import torch


def one_pole_lowpass(x, a):
    """ One pole filter. """
    a = torch.clamp(a, 0, 1)
    y = torch.zeros_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y


def one_pole_highpass(x, a):
    """ High-pass one-pole filter using matrix operations. """
    a = torch.clamp(a, 0, 1)
    y = torch.zeros_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y


def one_pole_bandpass(x, low_a, high_a):
    """ Band-pass filter by combining low-pass and high-pass filters. """
    low_passed = one_pole_lowpass(x, low_a)
    band_passed = one_pole_highpass(low_passed, high_a)
    return band_passed


def resonant_filter(x, cutoff, resonance):
    """ Resonant filter using matrix operations.
    Vectorized input contribution: (x[n] - b * x[n-1])
    Two Pole Formula: y[n] = a * (x[n] - b * x[n-1]) + c * y[n-1]
    """
    cutoff = torch.clamp(cutoff, 0, 1)
    resonance = torch.clamp(resonance, 0, 1)

    alpha = (1 - cutoff)
    beta = resonance

    # This is like a high-pass filtering step.
    input_diff = x - beta * torch.cat([x[:1], x[:-1]])

    output = torch.zeros_like(x)
    output[0] = input_diff[0]
    output[1:] = alpha * torch.cumsum(input_diff[1:], dim=0)
    return output


def dc_offset_filter(x):
    """ Remove the DC (mean) offset from the signal. """
    dc_component = torch.mean(x)
    return x - dc_component


def exponential_envelope(num_samples, attack_time, decay_time):
    """ Differentiable exponential envelope. """
    time = torch.linspace(0, 1, num_samples)
    attack_phase = time <= attack_time
    decay_phase = time > attack_time

    # Attack: sharp rise
    attack_envelope = (1 - torch.exp(-time / attack_time)) * attack_phase.float()

    # Decay: smooth fall
    decay_envelope = torch.exp(-(time - attack_time) / decay_time) * decay_phase.float()

    # Combine attack and decay
    envelope = (attack_envelope + decay_envelope)
    return envelope
