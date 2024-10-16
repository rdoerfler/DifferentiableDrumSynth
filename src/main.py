import soundfile as sf
import torch
import torch.nn as nn
import config
import matplotlib.pyplot as plt

from loss import DrumSoundLoss
from parametric_drum_synth import ParametricDrumSynth, logits_to_params

""" Train the drum synth model. """

if __name__ == '__main__':

    # Target audio
    file_name = 'ringing_snare'
    audio, sample_rate = sf.read(f'../inputs/{file_name}.wav')
    target_audio = torch.tensor(audio, dtype=torch.float32)
    num_samples = target_audio.shape[0]

    # Test DrumSynthLayer
    drum_synth = ParametricDrumSynth(sample_rate, num_samples)

    # Initialise Logits
    scaling_factors = config.scaling_factors()
    num_params = scaling_factors.size(0)
    emb = nn.Parameter(torch.randn(num_params))

    # Loss function
    loss_fn = DrumSoundLoss(sample_rate)

    losses = []

    net = torch.nn.Sequential(
        nn.Linear(num_params, 6),
        nn.ReLU(),
        nn.Linear(6, num_params),
        nn.ReLU()
    )

    lre = torch.linspace(-3, 0, 1000)
    lrs = 10 ** lre

    # Optimizer
    optimizer = torch.optim.Adam(list(net.parameters()) + [emb], lr=0.015)
    print(sum(p.nelement() for p in net.parameters()) + emb.nelement())

    for i in range(1000):
        # Forward
        optimizer.zero_grad()
        logits = net(emb)
        params = logits_to_params(logits, scaling_factors)
        audio_output = drum_synth(params)
        loss = loss_fn(audio_output[None, None, :], target_audio[None, None, :])

        # Backward
        loss.backward()
        optimizer.step()
        print(f"Epoch {i + 1}, Loss: {loss.item()}")

        losses.append(loss.item())

        # NaN Protection
        if loss.item() != loss.item():
            break

        # Write file on improvements
        losses.append(loss.item())
        if loss.item() <= min(losses):
            # Normalize and save audio
            audio_output /= torch.max(torch.abs(audio_output))
            sf.write(f'../outputs/{file_name}_pred.wav', audio_output.detach().numpy(), samplerate=sample_rate)

    # Plot loss curve
    plt.plot(losses)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
