# Denoising Diffusion Probabilistic Models

![](https://github.com/masslightsquared/understanding_diffusion_probabilistic_models/blob/main/denoising_dpm/images/celeba.gif)

In this repository, we implement the [Denoising Diffusion Probabilstic Models](https://arxiv.org/abs/2006.11239), a seminal paper in the diffusion model literature using PyTorch.

Diffusion Probabilistic Models (DPMs) are a type of probabilistic graphical model used for approximating distributions over high-dimensional spaces. They are parameterized Markov chains trained using variational inference to produce samples matching the data after finite time. The main advantage of using DPMs is that they can represent complex distributions and can easily handle multimodal distributions with overlapping modes.

**tl;dr**: Train a denoising model conditioned on the amount of noise present in the image, and generate samples by iteratively denoising from pure noise to a final sample.
