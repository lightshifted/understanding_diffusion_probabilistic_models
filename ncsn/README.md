# Denoising Score Matching with Langevin Dynamics (SMLD)

In this repository, we implement Noise Conditional Score Networks in PyTorch, as elucidated in [Generative Modeling by Estimating Gradients of the Data Distribution](http://arxiv.org/abs/1907.05600), a seminal paper in the diffusion model literature.

## To-Do: 
1. Implement NCSN improvements proposed in [Improved Techniques for Training Score-Based Generative Models](http://arxiv.org/abs/2006.09011)
2. Include connections to TPU-VM for generation of high resolution images (more pleasing aesthetically than 32x32!)
3. Ensure code is aligned with sound engineering principals (e.g. typing for functions, styling, etc.)
