import soundfile as sf
import torch
import config
from loss import DrumSoundLoss
from parametric_drum_synth import ParametricDrumSynth, logits_to_params

""" Train the drum synth model. """

if __name__ == '__main__':

    # Target audio
    file_name = 'hi_snare'
    audio, sample_rate = sf.read(f'../inputs/{file_name}.wav')
    target_audio = torch.tensor(audio, dtype=torch.float32)
    num_samples = target_audio.shape[0]

    # Test DrumSynthLayer
    drum_synth = ParametricDrumSynth(sample_rate, num_samples)

    # Initialise Logits
    logits = torch.randn(13, requires_grad=True)
    scaling_factors = config.scaling_factors()

    # Optimizer
    optimizer = torch.optim.Adam([logits], lr=0.1)

    # Loss function
    loss_fn = DrumSoundLoss(sample_rate)

    for i in range(200):
        optimizer.zero_grad()
        params = logits_to_params(logits, scaling_factors)
        audio_output = drum_synth(params)
        loss = loss_fn(audio_output[None, None, :], target_audio[None, None, :])
        loss.backward()
        optimizer.step()
        print(f"Epoch {i + 1}, Loss: {loss.item()}")

    # Normalize and save audio
    audio_output /= torch.max(torch.abs(audio_output))
    sf.write(f'../outputs/{file_name}_pred.wav', audio_output.detach().numpy(), samplerate=sample_rate)
