import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularDrumSynth(nn.Module):
    def __init__(self, max_delay):
        super().__init__()
        self.max_delay = max_delay

        # Resonator parameters
        self.res_decay = nn.Parameter(torch.tensor(0.95))
        self.res_freq = nn.Parameter(torch.tensor(440.0))
        self.res_buffer = nn.Parameter(torch.zeros(max_delay), requires_grad=False)

        # Transient parameters
        self.trans_attack = nn.Parameter(torch.tensor(0.01))
        self.trans_decay = nn.Parameter(torch.tensor(0.1))
        self.trans_gain = nn.Parameter(torch.tensor(1.0))

        # Noise parameters
        self.noise_gain = nn.Parameter(torch.tensor(0.5))
        self.noise_decay = nn.Parameter(torch.tensor(0.2))

    def forward(self, num_samples):
        # Resonator
        res_out = []
        current_buffer = self.res_buffer.clone()
        for _ in range(num_samples):
            delayed = current_buffer[0]
            current_buffer = torch.roll(current_buffer, shifts=-1)
            current_sample = self.res_decay * delayed
            current_buffer[-1] = current_sample
            res_out.append(current_sample)
        res_out = torch.stack(res_out)
        self.res_buffer.copy_(current_buffer)

        # Transient
        t = torch.linspace(0, 1, num_samples)
        trans_env = self.trans_gain * (torch.exp(-t / self.trans_attack) - torch.exp(-t / self.trans_decay))
        trans_out = trans_env * torch.randn(num_samples)

        # Noise
        noise_env = self.noise_gain * torch.exp(-t / self.noise_decay)
        noise_out = noise_env * torch.randn(num_samples)

        return res_out + trans_out + noise_out


class DirectParamDrumSynth(nn.Module):
    def __init__(self, max_delay):
        super().__init__()
        self.param_vector = nn.Parameter(torch.rand(8))  # 8 parameters for the drum synth
        self.drum_synth = ModularDrumSynth(max_delay)

    def forward(self, num_samples):
        # Apply appropriate scaling and constraints to the parameters
        params = torch.sigmoid(self.param_vector)

        self.drum_synth.res_decay.data = params[0]
        self.drum_synth.res_freq.data = params[1] * 1000  # Scale to 0-1000 Hz
        self.drum_synth.trans_attack.data = params[2] * 0.1  # Scale to 0-0.1 seconds
        self.drum_synth.trans_decay.data = params[3] * 0.5  # Scale to 0-0.5 seconds
        self.drum_synth.trans_gain.data = params[4]
        self.drum_synth.noise_gain.data = params[5]
        self.drum_synth.noise_decay.data = params[6] * 0.5  # Scale to 0-0.5 seconds

        # Generate audio
        return self.drum_synth(num_samples)


# Loss function
def multi_res_spectrum_loss(output, target):
    fft_sizes = [64, 128, 256, 512, 1024, 2048]
    loss = 0
    for size in fft_sizes:
        out_spec = torch.stft(output, size, return_complex=True, onesided=True, window=torch.hann_window(size))
        target_spec = torch.stft(target, size, return_complex=True, onesided=True, window=torch.hann_window(size))
        loss += F.mse_loss(out_spec.abs(), target_spec.abs(), reduction='mean')
    return loss


# Example usage

# Target audio
audio, sample_rate = sf.read('inputs/hi_snare.wav')
target_audio = torch.tensor(audio, dtype=torch.float32)
num_samples = target_audio.shape[0]

max_delay = 1000
model = DirectParamDrumSynth(max_delay)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    audio_output = model(num_samples=num_samples)
    loss = multi_res_spectrum_loss(audio_output, target_audio)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Generate audio
sf.write('outputs/drumhit.wav', audio_output.detach().numpy(), samplerate=sample_rate)