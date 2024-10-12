import soundfile as sf
import torch
import torch.nn as nn
import torch.functional as F
import auraloss

import config
from dsp_utils import one_pole_filter, exponential_envelope


class ParametricDrumSynth(nn.Module):
    """ ParametricDrumSynth class for differentiable drum sound synthesis.

    :param sample_rate: Sample rate of the audio.
    :param num_samples: Length of the audio.
    :returns audio_output: Audio output of the drum synth.
    """

    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        # Initialise Drum Synth Parameters
        self.transient_generator = TransientGenerator(self.sample_rate, self.num_samples)
        self.resonator = Resonator(self.sample_rate, self.num_samples)
        self.noise_generator = NoiseGenerator(self.sample_rate, self.num_samples)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """ Generates drum hit based on parameters. """

        transient = self.transient_generator(parameters[:5])
        resonance = self.resonator(transient, parameters[5:8])
        noise = self.noise_generator(parameters[8:])

        return transient + noise + resonance


class TransientGenerator(nn.Module):
    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """ Transient generator with envelope. """

        # Set parameters
        self.attack = parameters[0]
        self.decay = parameters[1]
        self.frequency = parameters[2]
        self.saturation = parameters[3]
        self.gain = parameters[4]

        # Generate audio
        sine_component = self._synthesize_sine(self.frequency, self.gain, self.decay)
        sine_component = torch.tanh(sine_component * self.saturation)
        noise_component = self._synthesize_noise(torch.tensor([0.1]), self.gain / 2, self.decay * 20)

        return sine_component + noise_component

    def _synthesize_sine(self, frequency: float, gain: float, decay: float) -> torch.Tensor:
        """ Sine wave with exponential decay. """

        time = torch.linspace(0, 1, self.num_samples)
        sine_wave = torch.sin(2 * torch.pi * frequency * time)
        decay = torch.exp(-decay * time)

        return sine_wave * decay * gain

    def _synthesize_noise(self, cutoff: float, gain: float, decay: float) -> torch.Tensor:
        """ Filtered noise with envelope. """

        time = torch.linspace(0, 1, self.num_samples)
        decay = torch.exp(-decay * time)

        noise = torch.randn(self.num_samples, requires_grad=False)
        noise = one_pole_filter(noise, cutoff)

        return noise * decay * gain


class Resonator(nn.Module):
    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super(Resonator, self).__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.register_buffer('time_indices', torch.arange(num_samples, dtype=torch.float32))

    def forward(self, transient: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """ Optimized differentiable resonator using efficient tensor operations. """

        frequency, feedback, gain = parameters[0], parameters[1], parameters[2]

        # Compute continuous delay in samples
        delay_samples = self.sample_rate / frequency

        # Compute delayed indices
        delayed_indices = self.time_indices - delay_samples

        # Create mask for valid delayed indices
        valid_mask = (delayed_indices >= 0).float()

        # Compute floor and ceil indices
        floor_indices = torch.floor(delayed_indices).long()
        ceil_indices = floor_indices + 1

        # Ensure indices are within bounds
        floor_indices = torch.clamp(floor_indices, 0, self.num_samples - 1)
        ceil_indices = torch.clamp(ceil_indices, 0, self.num_samples - 1)

        # Compute fractional part for interpolation
        frac = delayed_indices - floor_indices.float()

        # Initialize output tensor
        output = torch.zeros_like(transient)
        output[0] = transient[0]

        # Efficient computation using vectorized operations
        for i in range(1, self.num_samples):
            floor_value = output[floor_indices[i]]
            ceil_value = output[ceil_indices[i]]
            interpolated = (1 - frac[i]) * floor_value + frac[i] * ceil_value
            output[i] = transient[i] + feedback * interpolated * valid_mask[i]

        return output * gain


class NoiseGenerator(nn.Module):
    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super().__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """ Noise generator with envelope. """

        # Set parameters
        self.attack = parameters[0]
        self.decay = parameters[1]
        self.gain = parameters[2]

        # Generate audio
        envelope = exponential_envelope(self.num_samples, self.attack, self.decay)
        noise = torch.randn(self.num_samples, requires_grad=False)

        return noise * envelope * self.gain


def logits_to_params(logits: torch.Tensor, scaling_factors: torch.Tensor) -> torch.Tensor:
    """ Scales logits to synth parameters.

    Mapping:
    "transient_attack": 1.0,
    "transient_decay": 32,
    "transient_frequency": 250,
    "transient_saturation": 20,
    "transient_gain": 1.0,

    "resonator_frequency": 250,
    "resonator_feedback": 0.99,
    "resonator_gain": 1.0,

    "noise_attack": 1.0,
    "noise_decay": 32,
    "noise_gain": 1.0

    :param logits: Logits from the model.
    :param scaling_factors: Scaling factors for the logits.
    :returns parameters: Parameters for the drum synth.
    """

    parameters = torch.sigmoid(logits) * scaling_factors

    return parameters


if __name__ == '__main__':

    # Target audio
    audio, sample_rate = sf.read('../inputs/hi_snare.wav')
    target_audio = torch.tensor(audio, dtype=torch.float32)
    num_samples = target_audio.shape[0]

    # Test DrumSynthLayer
    drum_synth = ParametricDrumSynth(sample_rate, num_samples)
    for param in drum_synth.parameters():
        assert param.requires_grad == True

    # Get Logits (11 logits)
    logits = torch.randn(11, requires_grad=True)
    scaling_factors = config.scaling_factors()

    # Optimizer
    optimizer = torch.optim.Adam([logits], lr=0.01)

    # Loss function
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=sample_rate,
        perceptual_weighting=True,
    )

    for i in range(32):
        optimizer.zero_grad()
        params = logits_to_params(logits, scaling_factors)
        audio_output = drum_synth(params)
        loss = loss_fn(audio_output[None, None, :], target_audio[None, None, :])
        loss.backward()
        optimizer.step()
        print(f"Epoch {i + 1}, Loss: {loss.item()}")

    # Normalize and save audio
    audio_output /= torch.max(torch.abs(audio_output))
    sf.write('../outputs/drumhit.wav', audio_output.detach().numpy(), samplerate=sample_rate)
