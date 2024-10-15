import soundfile as sf
import torch
import torch.nn as nn
import config
from loss import DrumSoundLoss
from parametric_drum_synth import ParametricDrumSynth, logits_to_params

""" Train the drum synth model. """

if __name__ == '__main__':

    # Target audio
    file_name = 'frame_drum'
    audio, sample_rate = sf.read(f'../inputs/{file_name}.wav')
    target_audio = torch.tensor(audio, dtype=torch.float32)
    num_samples = target_audio.shape[0]

    # Test DrumSynthLayer
    drum_synth = ParametricDrumSynth(sample_rate, num_samples)

    # Initialise Logits
    scaling_factors = config.scaling_factors()
    num_params = scaling_factors.size(0)
    logits = nn.Parameter(torch.randn(num_params))

    # Optimizer
    optimizer = torch.optim.Adam([logits], lr=0.15)

    # Loss function
    loss_fn = DrumSoundLoss(sample_rate)

    losses = []
    for i in range(100):
        optimizer.zero_grad()
        params = logits_to_params(logits, scaling_factors)
        audio_output = drum_synth(params)
        loss = loss_fn(audio_output[None, None, :], target_audio[None, None, :])
        loss.backward()
        optimizer.step()
        print(f"Epoch {i + 1}, Loss: {loss.item()}")

        if loss.item() != loss.item():
            break

        losses.append(loss.item())

        if loss.item() <= min(losses):
            # Normalize and save audio
            audio_output /= torch.max(torch.abs(audio_output))
            sf.write(f'../outputs/{file_name}_pred.wav', audio_output.detach().numpy(), samplerate=sample_rate)
