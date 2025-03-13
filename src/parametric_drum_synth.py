import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dsp_utils import one_pole_lowpass, one_pole_highpass, exponential_envelope, one_pole_bandpass
from src.config import PARAM_MAP


class ParametricDrumSynth(nn.Module):
    """ParametricDrumSynth class for differentiable drum sound synthesis.

    Args:
        sample_rate: Sample rate of the audio in Hz
        chunk_size: Length of the audio in samples
    """

    def __init__(self, num_tones: int = 12, sample_rate: int = 48000, chunk_size: int = 24000):
        super().__init__()
        self.num_tones = num_tones
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Initialize Drum Synth Components
        self.transient_generator = TransientGenerator(self.sample_rate, self.chunk_size)
        self.tone_generator = ToneGenerator(self.num_tones, self.sample_rate, self.chunk_size)
        self.resonator = Resonator(self.sample_rate, self.chunk_size)
        self.noise_generator = NoiseGenerator(self.sample_rate, self.chunk_size)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Generates drum hit based on parameters.

        Args:
            params: Tensor of synthesizer parameters

        Returns:
            Synthesized audio output
        """
        # Apply small epsilon to avoid exactly zero parameters
        params = params.clamp(min=1e-7)

        # Organize params into dict of
        params = self._split_into_groups(params)

        # Generate components
        transient = self.transient_generator(params['transient_params'])
        tone = self.tone_generator(params['tone_params'])
        noise = self.noise_generator(params['noise_params'])
        mixed = (transient + tone) * 0.5
        resonance = self.resonator(mixed, params['resonator_params'])

        # Mix components with safety clipping to prevent overflow
        output = transient + tone + noise + resonance
        return torch.tanh(output)  # Soft clip to avoid extreme values

    @staticmethod
    def _split_into_groups(params: torch.Tensor):
        """Splits parameters into groups as defined in config."""
        params_mapped = {}
        idx = 0
        for name, val in PARAM_MAP.items():
            params_mapped[name] = params[idx:idx + val]
            idx += val
        return params_mapped


class TransientGenerator(nn.Module):
    """Generates transient (attack) portion of drum sound with envelope."""

    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.register_buffer('time', torch.linspace(0, 1, num_samples))

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Generate transient sound with enveloped sine and noise.

        Args:
            parameters: [attack, decay, frequency, saturation, gain]

        Returns:
            Transient audio component
        """
        # Apply safety clamping to parameters
        attack = parameters[0].clamp(min=1e-7)
        decay = parameters[1].clamp(min=1e-7)
        frequency = parameters[2].clamp(min=1.0)
        saturation = parameters[3].clamp(min=0.01, max=100.0)
        gain = parameters[4].clamp(min=1e-7)

        # Generate audio components
        sine_component = self._synthesize_sine(frequency, gain, decay)

        # Use tanh for soft saturation with gradient stability
        sine_component = torch.tanh(sine_component * saturation)

        # Generate noise with scaled parameters for stability
        noise_decay = torch.clamp(decay * 100, min=0.1, max=1000.0)

        # Synthesise transient noise
        noise_component = self._synthesize_noise(
            torch.tensor([0.1], device=parameters.device),
            gain / 10.0,
            noise_decay
        )

        return sine_component + noise_component

    def _synthesize_sine(
            self,
            frequency: torch.Tensor,
            gain: torch.Tensor,
            decay: torch.Tensor) -> torch.Tensor:
        """Generate sine wave with exponential decay.

        Args:
            frequency: Sine wave frequency in Hz
            gain: Amplitude scaling factor
            decay: Decay rate for exponential envelope

        Returns:
            Enveloped sine wave
        """
        phase = 2 * torch.pi * frequency * self.time
        sine_wave = torch.sin(phase)
        decay_env = torch.exp(-decay * self.time)

        return sine_wave * decay_env * gain

    def _synthesize_noise(
            self,
            cutoff: torch.Tensor,
            gain: torch.Tensor,
            decay: torch.Tensor) -> torch.Tensor:
        """Generate filtered noise with envelope.

        Args:
            cutoff: Lowpass filter cutoff frequency
            gain: Amplitude scaling factor
            decay: Decay rate for exponential envelope

        Returns:
            Filtered and enveloped noise
        """
        # Create noise with detached gradient to prevent NaN in backward pass
        noise = torch.randn(self.num_samples, device=gain.device)
        # Apply lowpass filtering
        filtered_noise = one_pole_lowpass(noise, cutoff)
        # Apply exponential decay envelope
        decay_env = torch.exp(-decay * self.time)

        return filtered_noise * decay_env * gain


class ToneGenerator(nn.Module):
    """Generates tonal (pitched) portion of drum sound."""

    def __init__(self, num_voices: int = 12, sample_rate: int = 48000, num_samples: int = 24000):
        super().__init__()
        self.num_voices = num_voices
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.register_buffer('time', torch.linspace(0, 1, num_samples))

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Generate tonal components from multiple sine oscillators.

        Args:
            parameters: Tensor containing [frequencies, decays, gains] for 4 oscillators

        Returns:
            Summed tonal audio output
        """
        # Split and safety clamp parameters
        frequencies = parameters[:self.num_voices].clamp(min=1.0)
        decays = parameters[self.num_voices:2 * self.num_voices].clamp(min=1e-7)
        gains = parameters[2 * self.num_voices:].clamp(min=1e-7)

        # Generate and sum tonal components
        sine_components = self._synthesize_tones(frequencies, gains, decays)
        return torch.sum(sine_components, dim=0)

    def _synthesize_tones(
            self,
            frequencies: torch.Tensor,
            gains: torch.Tensor,
            decays: torch.Tensor) -> torch.Tensor:
        """Generate multiple sine waves with exponential decay.

        Args:
            frequencies: Vector of frequencies for each oscillator
            gains: Vector of amplitude scaling factors
            decays: Vector of decay rates

        Returns:
            Tensor of enveloped sine waves (oscillators × samples)
        """
        # Reshape for broadcasting
        freq_expanded = frequencies.unsqueeze(1)
        decay_expanded = decays.unsqueeze(1)
        gain_expanded = gains.unsqueeze(1)

        # Generate phase and sine waves
        phase = 2 * torch.pi * freq_expanded * self.time
        sine_waves = torch.sin(phase)

        # Apply exponential decay envelope
        decay_env = torch.exp(-decay_expanded * self.time)

        # Apply gain scaling
        return sine_waves * decay_env * gain_expanded


class Resonator(nn.Module):
    """Applies resonant filtering to input signal."""

    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super(Resonator, self).__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.register_buffer('output_signal', torch.zeros(num_samples))
        self.register_buffer('delay_buffer', torch.zeros(num_samples))
        # Maximum number of delay lines to prevent excessive computation
        self.max_num_delays = 50

    def forward(self, input_signal: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Apply resonant filtering to input signal.

        Args:
            input_signal: Audio input to process
            parameters: [frequency, feedback, filter_low, filter_high, gain]

        Returns:
            Processed audio with resonance
        """
        # Unpack and safety clamp parameters
        frequency = parameters[0].clamp(min=20.0, max=self.sample_rate / 2)
        feedback = parameters[1].clamp(min=0.0, max=0.99)
        filter_low = parameters[2].clamp(min=1e-7, max=0.99)
        filter_high = parameters[3].clamp(min=1e-7, max=0.99)
        gain = parameters[4].clamp(min=1e-7)

        # Calculate delay in samples with safety bounds
        delay_length = torch.clamp(
            torch.floor(self.sample_rate / frequency),
            min=1,
            max=self.num_samples // 2
        ).long()

        # Calculate number of delays with upper limit
        num_delays = min(self.num_samples // delay_length, self.max_num_delays)

        # Create feedback powers with numerical stability
        feedback_powers = torch.pow(
            feedback,
            torch.arange(num_delays, device=input_signal.device, dtype=torch.float32)
        )

        # Initialize output with input
        output_signal = input_signal.clone()

        # Apply delays and feedback with improved numerical stability
        for i in range(1, num_delays):
            delay_start = i * delay_length
            if delay_start >= self.num_samples:
                break

            # Calculate delayed signal with scaling
            if self.num_samples - delay_start > 0:
                # Copy appropriate segment from delay buffer
                buffer_segment = self.delay_buffer[:self.num_samples - delay_start]
                # Apply feedback scaling
                delay_signal = buffer_segment * feedback_powers[i]
                # Apply bandpass filtering
                delay_signal = one_pole_bandpass(delay_signal, filter_low, filter_high)
                # Add to output
                output_signal[delay_start:] += delay_signal

        # Update delay buffer for future calls (detached to prevent gradient accumulation)
        self.delay_buffer = input_signal.detach()

        # Apply gain with soft clipping for stability
        return torch.tanh(output_signal * gain)


class NoiseGenerator(nn.Module):
    """Generates noise component with envelope and filtering."""

    def __init__(self, sample_rate: int = 48000, num_samples: int = 24000):
        super().__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.register_buffer('time', torch.linspace(0, 1, num_samples))

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Generate filtered noise with envelope.

        Args:
            parameters: [attack, decay, filter_lp, filter_hp, gain]

        Returns:
            Filtered and enveloped noise
        """
        # Unpack and safety clamp parameters
        attack = parameters[0].clamp(min=1e-7)
        decay = parameters[1].clamp(min=1e-7)
        filter_lp = parameters[2].clamp(min=1e-7, max=0.99)
        filter_hp = parameters[3].clamp(min=1e-7, max=0.99)
        gain = parameters[4].clamp(min=1e-7)

        # Generate envelope
        envelope = exponential_envelope(self.num_samples, attack, decay)

        # Generate noise (detached to prevent NaN gradients)
        noise = torch.randn(self.num_samples, device=parameters.device)

        # Apply filtering
        filtered = one_pole_lowpass(noise, filter_lp)
        filtered = one_pole_highpass(filtered, filter_hp)

        # Apply envelope and gain
        return filtered * envelope * gain


def logits_to_params(logits: torch.Tensor, scaling_factors: torch.Tensor) -> torch.Tensor:
    """Converts logits to synthesizer parameters with improved numerical stability.

    Args:
        logits: Unconstrained parameters from model
        scaling_factors: Scaling factors for each parameter

    Returns:
        Constrained synthesizer parameters
    """
    # Apply sigmoid with numerical stability
    normalized = torch.sigmoid(logits.clamp(min=-50.0, max=50.0))

    # Add small epsilon to prevent exactly zero values
    normalized = normalized.clamp(min=1e-7, max=1.0)

    # Apply scaling with safety factor
    return normalized * scaling_factors
