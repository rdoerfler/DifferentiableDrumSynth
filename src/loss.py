import torch
import auraloss


class DrumSoundLoss(torch.nn.Module):
    """ Custom loss function for drum synthesis. """
    def __init__(self, sample_rate: int = 48000):
        super().__init__()

        # Perceptual frequency loss
        self.loss_stft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192, 16384],
            hop_sizes=[128, 512, 2048, 4096],
            win_lengths=[1024, 2048, 8192, 16384],
            scale="mel",
            n_bins=128,
            sample_rate=sample_rate,
            perceptual_weighting=True,
        )

        # Energy Envelope loss
        self.loss_energy = RMSLoss(
            window_size=1024,
            hop_size=512,
            sample_rate=sample_rate,
        )

    def forward(self, x, y):
        """ Compute custom loss. """
        loss_stft = self.loss_stft(x, y)
        loss_energy = self.loss_energy(x, y)
        return loss_stft + loss_energy


class RMSLoss(torch.nn.Module):
    """ Root Mean Square Loss. """

    def __init__(self, window_size, hop_size, sample_rate):
        super(RMSLoss, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def forward(self, x, y):
        """ Compute RMS loss. """
        x_rms = torch.sqrt(torch.mean(x ** 2, dim=1))
        y_rms = torch.sqrt(torch.mean(y ** 2, dim=1))

        loss = torch.mean((x_rms - y_rms) ** 2)
        return loss