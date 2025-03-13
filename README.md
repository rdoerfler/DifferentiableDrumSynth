### Differentiable Drum Synthesizer

A PyTorch-based differentiable drum synthesizer that can learn to mimic target drum sounds through gradient descent.

#### Overview
This repository contains a parametric drum synthesis engine implemented as a differentiable neural network. 
The synthesizer can generate synthetic drum sounds by optimizing its parameters to match target audio samples through backpropagation.

#### Components
The synthesizer consists of four main components:

1. Transient Generator: Creates the initial attack portion of drum sounds using enveloped sine waves and noise
2. Tone Generator: Produces the pitched components using multiple sine oscillators with independent frequencies and decay rates
3. Resonator: Applies resonant filtering to enhance tonal characteristics
4. Noise Generator: Adds filtered noise with configurable envelope and filtering

#### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drum-synthesizer.git
cd drum-synthesizer

# Install dependencies
pip install -r requirements.txt
```

#### Usage
##### Training the Synthesizer
To train the synthesizer to match a target drum sound:

```python
import torch
import soundfile as sf
from parametric_drum_synth import ParametricDrumSynth, logits_to_params
from loss import DrumSoundLoss

# Load target audio
target_audio, sample_rate = sf.read('inputs/kick_drum.wav')
target_audio = torch.tensor(target_audio, dtype=torch.float32)

# Initialize the synthesizer
synth = ParametricDrumSynth(sample_rate=sample_rate, chunk_size=len(target_audio))

# Train the model
from train_drum_synth import train_drum_synth
best_logits, loss_history = train_drum_synth('kick_drum', epochs=1000)

# Generate the final sound
scaling_factors = config.scaling_factors()
params = logits_to_params(best_logits, scaling_factors)
audio_output = synth(params)

# Save output
sf.write('outputs/kick_drum_synth.wav', audio_output.detach().numpy(), sample_rate)
```

##### Using Pre-trained Parameters

```python 
import torch
import soundfile as sf
from parametric_drum_synth import ParametricDrumSynth

# Load model checkpoint
checkpoint = torch.load('outputs/kick_drum_best_model.pt')
params = checkpoint['params']

# Initialize synthesizer with saved parameters
synth = ParametricDrumSynth(sample_rate=48000)
audio_output = synth(params)

# Save output
sf.write('outputs/kick_drum_playback.wav', audio_output.detach().numpy(), 48000)
```

#### License
This project is licensed under the MIT License - see the LICENSE file for details.
#### Acknowledgments

This project was inspired by research in differentiable digital signal processing (DDSP).
MRSTFT Loss function based on auraloss by Christian Steinmetz et al. 