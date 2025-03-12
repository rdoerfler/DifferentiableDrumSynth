import torch
import soundfile as sf
from src.parametric_drum_synth import Resonator
from src.config import scaling_factors
import pytest

SAMPLE_RATE = 48000
CHUNK_SIZE = 96000


@pytest.fixture()
def resonator(sample_rate: int = 48000, chunk_size: int = 96000):
    return Resonator(sample_rate=SAMPLE_RATE, num_samples=CHUNK_SIZE)


def test_w_random_inputs(resonator):
    # Random Logits
    logits = torch.randn(5, requires_grad=True)
    logits = torch.sigmoid(logits) * scaling_factors()[5:10]

    # Test Signal
    stimulus = torch.rand(CHUNK_SIZE, requires_grad=False) * torch.linspace(1, 0, CHUNK_SIZE) ** 10

    # Test Resonator
    audio_output = resonator(stimulus, logits)

    # Normalize and save audio
    audio_output /= torch.max(torch.abs(audio_output))
    sf.write(f'../outputs/resonator_test.wav', audio_output.detach().numpy(), samplerate=SAMPLE_RATE)
