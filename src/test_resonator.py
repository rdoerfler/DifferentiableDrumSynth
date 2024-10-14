import torch
import soundfile as sf

from parametric_drum_synth import Resonator

# Initialise Resonator
sample_rate = 48000
num_samples = 96000
resonator = Resonator(sample_rate=sample_rate, num_samples=num_samples)

# Random Logits
logits = torch.randn(3, requires_grad=True)
logits = torch.sigmoid(logits) * torch.tensor([250, 0.999, 1.0])

# Test Signal
stimulus = torch.rand(num_samples, requires_grad=False) * torch.linspace(1, 0, num_samples) ** 10

# Test Resonator
audio_output = resonator(stimulus, logits)

# Normalize and save audio
audio_output /= torch.max(torch.abs(audio_output))
sf.write(f'../outputs/resonator_test.wav', audio_output.detach().numpy(), samplerate=sample_rate)
