import torch
import auraloss


class DrumSoundLoss(torch.nn.Module):
    """ Custom loss function for drum synthesis. """
    def __init__(self, sample_rate: int = 48000):
        super().__init__()

        fft_sizes = [16384, 8192, 4096, 2048, 1024, 512, 128, 32]
        overlap = 0.75
        hop_sizes = [int(i * (1 - overlap)) for i in fft_sizes]

        # Perceptual frequency loss
        self.loss_stft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=fft_sizes,
            scale=None,
            n_bins=128,
            sample_rate=sample_rate,
            perceptual_weighting=True,
            w_log_mag=1.0,
            w_sc=1.0
        )

        # Energy Envelope loss
        self.loss_energy = RMSLoss()

    def forward(self, x, y):
        """ Compute custom loss. """
        stft_loss = self.loss_stft(x, y)
        energy_loss = self.loss_energy(x, y)
        return stft_loss + energy_loss * 0.25


class RMSLoss(torch.nn.Module):
    """ Root Mean Square Loss. """

    def __init__(self):
        super(RMSLoss, self).__init__()

    def forward(self, x, y):
        """ Compute RMS loss. """
        x_rms = torch.sqrt(torch.mean(x ** 2, dim=1))
        y_rms = torch.sqrt(torch.mean(y ** 2, dim=1))

        loss = torch.mean((x_rms - y_rms) ** 2)
        return loss
