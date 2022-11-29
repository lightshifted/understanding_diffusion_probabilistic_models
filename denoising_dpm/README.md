# Denoising Diffusion Probabilistic Models

![](https://github.com/masslightsquared/understanding_diffusion_probabilistic_models/blob/main/denoising_dpm/images/celeba.gif)

## 1. Project Overview
In this repository, we implement the [Denoising Diffusion Probabilstic Models](https://arxiv.org/abs/2006.11239), a seminal paper in the diffusion model literature using PyTorch. This implementation is for developing a deeper understanding of the class of models core to Stable Diffusion. It was designed not for production, but for exploration and increasing intution of diffusion probabilistic models. 

Diffusion Probabilistic Models (DPMs) are a type of probabilistic graphical model used for approximating distributions over high-dimensional spaces. They are parameterized Markov chains trained using variational inference to produce samples matching the data after finite time. The main advantage of using DPMs is that they can represent complex distributions and can easily handle multimodal distributions with overlapping modes.

**tl;dr**: Train a denoising model conditioned on the amount of noise present in the image, and generate samples by iteratively denoising from pure noise to a final sample.

## 2. Installation & Requirements
**With `pip`**
```bash
python3 -m pip install -e . # installs required packages only
```
PyTorch $\geq$ 1.13.0 and Python $\geq$ 3.8.5 are required.

## 3. Getting Started
Check the `ddpm_example` notebook for an example use of the codebase.

## 4. Resources
1. Denoising [Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) paper by Yang Song and team.
2. The excellent blog post [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) by Yang Song.
