import torch
import auraloss

# define the loss function
loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    scale="mel",
    n_bins=128,
    sample_rate=sample_rate,
    perceptual_weighting=True,
)