import torch

MODEL_PARAMS = dict(
    epochs=1000,
    learning_rate=1e-3,
)


num_tones = 4

PARAM_MAP = dict(
    transient_params=5,
    tone_params=num_tones * 3,
    resonator_params=5,
    noise_params=5,
)


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
        "transient_decay": 64,
        "transient_frequency": 600,
        "transient_saturation": 12,
        "transient_gain": 1.0,

        "tone_frequency_01": 400,
        "tone_frequency_02": 400,
        "tone_frequency_03": 400,
        "tone_frequency_04": 400,
        "tone_decay_01": 32,
        "tone_decay_02": 32,
        "tone_decay_03": 32,
        "tone_decay_04": 32,
        "tone_gain_01": 1.0,
        "tone_gain_02": 1.0,
        "tone_gain_03": 1.0,
        "tone_gain_04": 1.0,

        "resonator_frequency": 250,
        "resonator_feedback": 0.999,
        "resonator_filter_lp": 1.0,
        "resonator_filter_hp": 1.0,
        "resonator_gain": 1.0,

        "noise_attack": 0.01,
        "noise_decay": 0.5,
        "noise_filter_lp": 1.0,
        "noise_filter_hp": 1.0,
        "noise_gain": 0.01
    }

    return torch.tensor(list(scalers.values()), requires_grad=False)
