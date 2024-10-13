import torch
import torch.nn as nn
from dsp_utils import one_pole_lowpass, one_pole_highpass, exponential_envelope, resonant_filter, dc_offset_filter


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
        # self.resonator = Resonator(self.sample_rate, self.num_samples)
        self.noise_generator = NoiseGenerator(self.sample_rate, self.num_samples)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """ Generates drum hit based on parameters. """

        transient = self.transient_generator(parameters[:5])
        # resonance = self.resonator(transient, parameters[5:8])
        noise = self.noise_generator(parameters[8:])

        return transient + noise # + resonance


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
        noise = one_pole_lowpass(noise, cutoff)

        return noise * decay * gain


class Resonator(nn.Module):
    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        """
        Args:
            sample_rate: The sampling rate of the audio (default is 48 kHz).
            num_samples: Number of samples to process in one forward pass.
        """
        super(Resonator, self).__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate

        # Time indices for all samples, used for computing delay
        self.register_buffer('output_signal', torch.zeros(num_samples, requires_grad=False))
        self.register_buffer('delay_buffer', torch.zeros(num_samples, requires_grad=False))

    def forward(self, transient: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transient: 1D input signal of shape (num_samples).
            parameters: A tensor containing the control parameters [frequency, feedback, gain].
                        - frequency: Controls delay length based on the frequency.
                        - feedback: Controls the feedback loop (0 to 1).
                        - gain: Scales the output signal.
        Returns:
            output_signal: 1D output signal after applying the resonator.
        """

        frequency, feedback, gain = parameters[0], parameters[1], parameters[2]

        delay_samples = self.sample_rate / frequency

        # Compute the delay line
        self.delay_buffer = torch.zeros(self.num_samples, requires_grad=False)
        return self.output_signal


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
