import soundfile as sf
import torch
import torch.nn as nn
import config
from loss import DrumSoundLoss
from parametric_drum_synth import ParametricDrumSynth, logits_to_params
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

""" Train the drum synth model. """


def train_drum_synth(file_name, epochs=1000, lr=0.1, device=None):
    # Use GPU if available and not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure output directory exists
    os.makedirs('../outputs', exist_ok=True)

    # Target audio
    audio, sample_rate = sf.read(f'../inputs/{file_name}.wav')
    target_audio = torch.tensor(audio, dtype=torch.float32).to(device)
    num_samples = target_audio.shape[0]

    # Initialize model
    drum_synth = ParametricDrumSynth(sample_rate, num_samples).to(device)

    # Initialize parameters
    scaling_factors = config.scaling_factors().to(device)
    num_params = scaling_factors.size(0)
    logits = nn.Parameter(torch.randn(num_params, device=device))

    # Optimizer
    optimizer = torch.optim.Adam([logits], lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )

    # Loss function
    loss_fn = DrumSoundLoss(sample_rate).to(device)

    # Prepare target audio for batch processing
    target_audio_batch = target_audio[None, None, :]

    # Initialize tracking variables
    losses = []
    best_loss = float('inf')

    # Training loop
    for i in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()

        # Forward pass
        params = logits_to_params(logits, scaling_factors)
        audio_output = drum_synth(params)
        audio_output_batch = audio_output[None, None, :]

        # Calculate loss
        loss = loss_fn(audio_output_batch, target_audio_batch)

        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN encountered at epoch {i + 1}. Stopping training.")
            break

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update scheduler
        scheduler.step(loss)

        # Track loss
        current_loss = loss.item()
        losses.append(current_loss)

        # Periodic logging
        if i % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses[-10:]))
            print(f"Epoch {i + 1}, Loss: {current_loss:.6f}, Avg Loss: {avg_loss:.6f}")

        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({
                'epoch': i,
                'logits': logits.detach().cpu(),
                'loss': current_loss,
                'params': params.detach().cpu()
            }, f'../outputs/{file_name}_best_model.pt')

            # Save audio for best model
            if i % 50 == 0 or i == epochs - 1:
                normalized_audio = audio_output.detach().cpu()
                normalized_audio /= torch.max(torch.abs(normalized_audio))
                sf.write(f'../outputs/{file_name}_pred.wav',
                         normalized_audio.numpy(),
                         samplerate=sample_rate)

    # Final result
    print(f"Training completed. Best loss: {best_loss:.6f}")
    return logits.detach().cpu(), losses


if __name__ == '__main__':
    # hyper params
    epochs = 1000
    lr = 1e-3

    # target file
    file_name = 'conga'

    # fit model
    best_logits, loss_history = train_drum_synth(file_name, epochs, lr)

    # plot loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'../outputs/{file_name}_loss_curve.png')
    plt.close()

