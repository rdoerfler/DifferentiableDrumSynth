import torch
import torch.nn as nn

from dsp_utils import one_pole_lowpass, one_pole_highpass, exponential_envelope, one_pole_bandpass


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
        self.tone_generator = ToneGenerator(self.sample_rate, self.num_samples)
        self.resonator = Resonator(self.sample_rate, self.num_samples)
        self.noise_generator = NoiseGenerator(self.sample_rate, self.num_samples)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """ Generates drum hit based on parameters. """

        transient = self.transient_generator(parameters[:5])
        tone = self.tone_generator(parameters[5:17])
        # resonance = self.resonator(transient + tone, parameters[5:10])
        noise = self.noise_generator(parameters[18:])

        return transient + tone + noise


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
        noise_component = self._synthesize_noise(torch.tensor([0.1]), self.gain / 10, self.decay * 100)

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
        noise = one_pole_lowpass(noise, cutoff)

        return noise * decay * gain


class ToneGenerator(nn.Module):
    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """ Transient generator with envelope. """

        # Set parameters
        self.frequencies = parameters[:4]
        self.decays = parameters[4:8]
        self.gains = parameters[8:12]

        # Generate audio
        sine_components = self._synthesize_tones(self.frequencies, self.gains, self.decays)

        return torch.sum(sine_components, dim=0)

    def _synthesize_tones(self, frequencies: float, gains: float, decays: float) -> torch.Tensor:
        """ Sine wave with exponential decay. """

        time = torch.linspace(0, 1, self.num_samples)
        sine_waves = torch.sin(2 * torch.pi * frequencies[:, None] * time)
        decays = torch.exp(-decays[:, None] * time)

        return sine_waves * decays * gains[:, None]


class Resonator(nn.Module):
    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super(Resonator, self).__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.register_buffer('output_signal', torch.zeros(num_samples))
        self.register_buffer('delay_buffer', torch.zeros(num_samples))

    def forward(self, input_signal: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        frequency, feedback, filter_low, filter_high, gain = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]

        # Calculate the delay in samples
        delay_length = (self.sample_rate / frequency).long()

        # Ensure delay_length is at least 1
        delay_length = max(delay_length, 1)

        # Calculate number of delays
        num_delays = self.num_samples // delay_length

        # Create a tensor of powers of feedback
        feedback_powers = torch.pow(feedback, torch.arange(num_delays, device=input_signal.device))

        # Initialize output tensor
        output_signal = input_signal.clone()

        # Apply delays and feedback
        for i in range(1, num_delays):
            delay_start = i * delay_length
            delay_signal = self.delay_buffer[:self.num_samples - delay_start] * feedback_powers[i]
            delay_signal = one_pole_bandpass(delay_signal, filter_low, filter_high)
            output_signal[delay_start:] += delay_signal

        return output_signal * gain


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
        self.filter_lp = parameters[2]
        self.filter_hp = parameters[3]
        self.gain = parameters[4]

        # Generate audio
        envelope = exponential_envelope(self.num_samples, self.attack, self.decay)
        noise = torch.randn(self.num_samples, requires_grad=False)
        noise = one_pole_lowpass(noise, self.filter_lp)
        noise = one_pole_highpass(noise, self.filter_hp)
        return noise * envelope * self.gain


def logits_to_params(logits: torch.Tensor, scaling_factors: torch.Tensor) -> torch.Tensor:
    """ Scales logits to synth parameters.

    :param logits: Logits from the model.
    :param scaling_factors: Scaling factors for the logits.
    :returns: Sigmoid function applied to parameters.
    """

    return (torch.sigmoid(logits) ** 1) * scaling_factors
