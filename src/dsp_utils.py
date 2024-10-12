import torch


def one_pole_filter(x, a):
    """ One pole filter. """
    a = torch.clamp(a, 0, 1)

    y = torch.zeros_like(x)
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = x[n] - a * x[n - 1]

    return y


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
