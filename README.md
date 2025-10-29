# Image Classification with Diffusion Equation

**Authors:** Ani Okropiridze, Mariam Mamageishvili  
**Supervisors:** Prof. Ramaz Botchorishvili, Dr. Hanno Scharr  
**Affiliation:** Kutaisi International University, School of Mathematics  
**Year:** 2025  

---

## Overview

This project explores a **PDE-based approach to neural networks**, replacing traditional convolutional layers with diffusion processes. We model image feature extraction as a **time-evolving diffusion process** with learnable coefficients \(\alpha(t, y)\) and \(\beta(t, x)\). The method is validated on grayscale and RGB image datasets such as MNIST, Fashion-MNIST, CIFAR-10, and SVHN.

---

## Key Concepts

- **PDE Formulation:** 2D diffusion equation with variable coefficients

\[
\frac{\partial u}{\partial t} = \alpha(t, y)\frac{\partial^2 u}{\partial x^2} + \beta(t, x)\frac{\partial^2 u}{\partial y^2}
\]

- **Initial & Boundary Conditions:** Supports zero or image-based Dirichlet boundaries.
- **Operator Splitting:** Strang splitting decomposes the 2D PDE into sequential 1D solves for efficiency.
- **Spatial Discretization:** Uniform grids with second-order central differences.
- **Temporal Discretization:** Implicit Euler for stability; also tested explicit schemes for efficiency.

---

## Implementation Details

- **Framework:** PyTorch
- **Diffusion Layer:** Learnable parameter tensors (\(\alpha, \beta\))
- **Feature Processing:** Flattening and dropout layers
- **Classification Head:** Fully connected layers
- **Optimizer & Loss:** Adam optimizer with Cross-Entropy loss
- **Numerical Methods:** Strang splitting, Thomas algorithm for tridiagonal systems

---

## Experimental Results

### MNIST (Grayscale)
- Test accuracy: **97.79%**
- Parameter efficiency: scalar coefficients performed comparably to full matrices.

### Fashion-MNIST (Grayscale)
- Test accuracy: **86.98%**

### SVHN (RGB digits)
- Test accuracy: **82.52%**
- Shows good performance on structured geometric patterns.

### CIFAR-10 (RGB natural images)
- Test accuracy: **51.23%**
- Lower performance due to complexity of natural textures.

**Observation:** PDE-based networks excel in structured grayscale or simple RGB data but face challenges with complex natural images.

---

## Future Work

- Explore **non-linear, time-dependent** diffusion coefficients.
- Model \(\alpha\) and \(\beta\) using **neural networks** or **Fourier series representations**.
- Apply **mesh refinement** and higher-order discretization methods.
- Improve **RGB image classification** by coupling PDE-based layers with advanced architectures.

---

## References

1. Ruthotto, L., & Haber, E. (2018). **Deep Neural Networks Motivated by Partial Differential Equations**. [arXiv:1804.01527](https://arxiv.org/abs/1804.01527)
2. [Strang Splitting for PDEs](https://arxiv.org/html/2408.06655v1)


