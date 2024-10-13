import torch


def test_logits():
    logits = {
        "transient_attack": 0.01,
        "transient_decay": 0.8,
        "transient_frequency": 0.5,
        "transient_saturation": 0.5,
        "transient_gain": 1.0,

        "resonator_frequency": 0.01,
        "resonator_feedback": 0.95,
        "resonator_gain": 0.808,

        "noise_attack": 0.01,
        "noise_decay": 0.3,
        "noise_filter": 0.5,
        "noise_gain": 0.5
    }

    return torch.tensor(list(logits.values()))


def scaling_factors():
    scalers = {
        "transient_attack": 1.0,
        "transient_decay": 32,
        "transient_frequency": 1200,
        "transient_saturation": 32,
        "transient_gain": 1.0,

        "resonator_frequency": 600,
        "resonator_feedback": 0.999,
        "resonator_gain": 1.0,

        "noise_attack": 0.01,
        "noise_decay": 0.5,
        "noise_filter_lp": 1.0,
        "noise_filter_hp": 1.0,
        "noise_gain": 0.3
    }

    return torch.tensor(list(scalers.values()), requires_grad=False)
