# AGENTS.md

This file provides guidance to Agents when working with code in this repository.

## Repository Overview

This repository contains course materials for "Modern Machine Learning Models" (Современные Модели Машинного Обучения - СММО). 
The course focuses on diffusion models and the HuggingFace ecosystem, with materials primarily in Russian.

## Repository Structure

```
ml-modern/
├── doc/              # Lecture and seminar notes (Markdown)
├── practice/         # Practice Jupyter notebooks by seminar
│   ├── sem01/       # Practical implementation of diffusion models
│   ├── sem02/       # Working with HuggingFace library
│   └── sem03/       # Diffusion models with Accelerate library
├── homework/         # Homework assignments
│   └── hw01/        # Latent diffusion model implementation
└── 2025_Современные_модели_машинного_обучения_силлабус.pdf
```

## Key Technologies and Libraries

All notebooks are designed to run in **Google Colab** and use:

- **PyTorch** - Deep learning framework
- **HuggingFace Diffusers** - Pre-trained diffusion models and pipelines
- **HuggingFace Datasets** - Dataset loading and processing
- **HuggingFace Accelerate** - Distributed training and optimization
- **einops** - Tensor operations
- **matplotlib** - Visualization
- **tqdm** - Progress bars

# Context Rules

## Constraints
NEVER use system-wide python. Use venvs managed by uv instead.
NEVER use icons or emojis in the code.
ALWAYS check if the package is installed before installing it.
DO NOT use uppercase words in the code, filenames, variables, functions, classes, etc. ONLY when it is required.

## Standards
Always use context7 when:
- Writing setup or configuration code.
- Generating or debugging examples.
- Accessing library or SDK documentation.

Use venvs managed by uv. Uv directory for venvs is ~/.cache/uv/virtualenvs.

## Automation
[[calls]]
match = "when the user requests library setup, API examples, or documentation"
tool = "context7"


## Working with Jupyter Notebooks

### Installing Dependencies

Notebooks typically include installation commands at the beginning:
```python
!pip install -q -U einops datasets matplotlib tqdm numpy torch torchvision diffusers accelerate
```

### Common Imports

Standard import pattern from the notebooks:
```python
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
```


## Course Topics

### Module 1: Diffusion Models (Семинар 1)
- Practical implementation of DDPM (Denoising Diffusion Probabilistic Models)
- U-Net architecture for diffusion models
- Forward and reverse diffusion processes
- Noise scheduling (cosine, linear)

**Practice Notebook**: [practice/sem01/СММО_Семинар_1_Практическая_реализация_диффузионной_модели.ipynb](practice/sem01/СММО_Семинар_1_Практическая_реализация_диффузионной_модели.ipynb)

### Module 2: HuggingFace Library (Семинар 2)
- HuggingFace pipelines for diffusion models
- Model loading and inference
- Scheduler configuration
- GPU vs CPU execution
- Model tracking with Weights & Biases and Neptune.ai

**Practice Notebook**: [practice/sem02/СММО_Семинар_2_Работа_с_библиотекой_HuggingFace_1.ipynb](practice/sem02/СММО_Семинар_2_Работа_с_библиотекой_HuggingFace_1.ipynb)

### Module 3: Distributed Training with Accelerate (Семинар 3)
- HuggingFace Accelerate for multi-GPU training
- Distributed training across multiple nodes
- Mixed precision training
- Gradient accumulation

**Practice Notebook**: [practice/sem03/СММО_Семинар_3_Изучение_особенностей_диффузионных_моделей_с_библиотекой_accelerate.ipynb](practice/sem03/СММО_Семинар_3_Изучение_особенностей_диффузионных_моделей_с_библиотекой_accelerate.ipynb)

## Homework Assignments

### Homework 1: Latent Diffusion Model
**Dataset**: `nelorth/oxford-flowers` from HuggingFace
**Tasks**:
1. Implement diffusion model training on the Oxford Flowers dataset
2. Implement latent diffusion model using VQ-VAE encoder
   - Use pre-trained VAE from Stable Diffusion: `AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")`
   - Encode images to latent space before applying diffusion process
   - Adjust U-Net architecture for smaller latent dimensions

**Expected Deliverables**:
- Training code
- Visualizations of generated images
- Written conclusions (mandatory): "Я хотел сделать вот это, были такие проблемы. Вот что вышло."

**Homework Notebook**: [homework/hw01/hw01.ipynb](homework/hw01/hw01.ipynb)

## Architecture Notes

### Diffusion Models Key Concepts

1. **Forward Process (q)**: Gradually adds Gaussian noise to images
   - Must use isotropic (independent) Gaussian noise
   - Noise schedule determines how noise is added over timesteps

2. **Reverse Process (p_θ)**: Neural network learns to denoise images
   - U-Net architecture with downsampling and upsampling paths
   - Skip connections between encoder and decoder

3. **Latent Diffusion**:
   - Apply diffusion in latent space instead of pixel space
   - Significantly reduces computational requirements
   - Requires VQ-VAE or similar encoder/decoder

### U-Net Architecture
- **Downsampling path**: Reduces spatial dimensions, increases channels
- **Upsampling path**: Increases spatial dimensions, reduces channels
- **Skip connections**: Concatenate features from downsampling to upsampling
- Modern implementations use Neural Architecture Search (NAS)

### Noise Schedulers
Pre-implemented schedulers are available in HuggingFace Diffusers:
- Cosine schedule (recommended for better performance)
- Linear schedule
- Customizable schedules

## Notes and Documentation

The [doc/](doc/) directory contains:
- Lecture notes with embedded images: `lec01.md`, `lec01 Gen models.md`
- Seminar session notes: `2025-10-13 sem01.md`, `2025-10-14 sem02.md`, `2025-10-20 sem03 Accelerate.md`
- General notes: `Notes.md`, `Info.md`
- Reference to [International Conference on Machine Learning (ICML)](https://icml.cc/)

**Note**: These files contain references to images stored in `.obsidian/` directory (Obsidian vault format).

## Language Note

All course materials (notebooks, notes, comments) are in **Russian**. When working with this repository:
- Keep code comments in Russian to maintain consistency
- Variable names and function names follow standard English programming conventions
- User-facing text and documentation should be in Russian
