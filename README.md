# CNN-with-PDE: PDE-Inspired Image Classification Without Convolutions

**Authors:** Ani Okropiridze, Mariam Mamageishvili  
**Supervisors:** Prof. Ramaz Botchorishvili, Dr. Hanno Scharr  
**Affiliation:** Kutaisi International University, School of Mathematics  
**Year:** 2025

## Project Summary

This repository studies an alternative to standard convolutional feature extraction: instead of applying learned convolution kernels, the model evolves an input image under a **learnable diffusion process** derived from a parabolic partial differential equation (PDE), and then classifies the transformed representation with fully connected layers.

The core research idea is:

1. Treat an image as an initial condition $u(0,x,y)$.
2. Evolve it for a small artificial time horizon $t \in [0,T]$.
3. Learn the diffusion coefficients that control how smoothing happens across space and time.
4. Use the evolved state \(u(T,x,y)\) as the feature representation for classification.

This makes the project relevant for AI research interviews because it connects:

- deep learning and scientific computing,
- inductive bias design,
- numerical PDE solvers inside differentiable models,
- stability-aware model construction,
- interpretable continuous-time feature dynamics.

## Why This Project Is Interesting

Most image classifiers rely on convolutions because convolutions learn local spatial filters efficiently. This project asks a different question:

**Can image features be learned through the dynamics of a discretized PDE rather than through learned convolution kernels?**

That question matters because PDE-based architectures provide:

- a mathematically grounded view of feature propagation,
- explicit control over smoothing, anisotropy, and stability,
- a bridge between continuous-time modeling and neural networks,
- interpretable learnable parameters with physical meaning.

In this repository, the main inductive bias is **anisotropic diffusion**. Instead of learning arbitrary filters, the model learns where and how strongly information should diffuse along the horizontal and vertical directions.

## Mathematical Formulation

The main model family is based on a 2D diffusion equation of the form

$$
\frac{\partial u}{\partial t}
=
\alpha(t,x,y)\,\frac{\partial^2 u}{\partial x^2}
+
\beta(t,x,y)\,\frac{\partial^2 u}{\partial y^2},
$$

where:

- $u(t,x,y)$ is the image intensity or feature value at artificial time $t$,
- $\alpha(t,x,y)$ controls diffusion in the $x$-direction,
- $\beta(t,x,y)$ controls diffusion in the $y$-direction.

In the code, these coefficients are learned as tensors. A common parameterization used in the repository is

$$
\alpha(t,x,y) = \alpha_{\text{base}}(x,y) + t\,\alpha_{\text{time}}(x,y),
\qquad
\beta(t,x,y) = \beta_{\text{base}}(x,y) + t\,\beta_{\text{time}}(x,y),
$$

followed by clamping to keep the coefficients positive:

$$
\alpha(t,x,y) \ge \varepsilon, \qquad \beta(t,x,y) \ge \varepsilon.
$$

This positivity constraint is important because it preserves the interpretation of the layer as a diffusion operator and improves numerical stability.

### Interpretation

- Large $\alpha$ means stronger horizontal smoothing.
- Large $\beta$ means stronger vertical smoothing.
- Spatially varying coefficients allow the model to smooth different regions differently.
- Time-dependent coefficients allow the dynamics to change as the feature map evolves.

So, rather than learning a bank of convolution kernels, the model learns a **spatiotemporal diffusion field**.

## From PDE to Neural Network Layer

For an input image $u^0$, a diffusion layer computes a sequence

$$
u^0 \rightarrow u^1 \rightarrow u^2 \rightarrow \cdots \rightarrow u^N,
$$

where each step approximates PDE evolution over a small time increment $\Delta t$.

The final state $u^N$ is flattened and passed to an MLP classifier. In the simplest grayscale case, the architecture is:

$$
\text{image} \rightarrow \text{PDE diffusion layer} \rightarrow \text{flatten} \rightarrow \text{dropout / MLP} \rightarrow \text{logits}.
$$

For RGB datasets, the repository extends this idea with:

- channel-wise coefficient fields,
- simple channel mixing / coupling,
- residual or skip connections in some variants,
- deeper fully connected heads,
- attention-style modules in the more experimental CIFAR models.

## Numerical Method

One of the strongest parts of this project is that the PDE layer is not just a metaphor. It is implemented as an actual numerical solver embedded inside PyTorch.

### 1. Operator Splitting

Several scripts use **Strang splitting** to decompose the 2D problem into 1D subproblems:

$$
u^{n+1}
\approx
S_x\left(\frac{\Delta t}{2}\right)
S_y(\Delta t)
S_x\left(\frac{\Delta t}{2}\right)u^n.
$$

This is attractive because solving two 1D diffusion systems is cheaper and more stable than solving the full 2D implicit system directly.

### 2. Spatial Discretization

Second derivatives are approximated with central finite differences. For example, in the $x$-direction:

$$
\frac{\partial^2 u}{\partial x^2}(x_i,y_j)
\approx
\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2}.
$$

Analogously for the $y$-direction:

$$
\frac{\partial^2 u}{\partial y^2}(x_i,y_j)
\approx
\frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}.
$$

### 3. Implicit Time Stepping

In the main diffusion-layer implementations, the resulting 1D diffusion problems are solved implicitly. For a 1D step, the update has the generic form

$$
(I - rA)u^{n+1} = u^n,
$$

where $A$ is the tridiagonal finite-difference operator and $r$ depends on the local diffusion coefficient and the ratio $\Delta t / \Delta x^2$ or $\Delta t / \Delta y^2$.

The advantage of the implicit scheme is improved stability compared with explicit diffusion updates, especially when the learned coefficients become large.

### 4. Batched Thomas Solver

Because the implicit 1D systems are tridiagonal, they can be solved efficiently with the **Thomas algorithm**. This repository implements batched tridiagonal solves so the PDE layer remains differentiable and GPU-compatible.

That is an important engineering contribution of the project: the model is not merely inspired by PDEs, but actually trains through a structured numerical solver.

### 5. Boundary Conditions

Most diffusion-layer variants use **Neumann-type no-flux boundary conditions**, implemented by modifying the first and last equations in the tridiagonal systems. Intuitively, this discourages artificial information loss at image borders.

## Repository Structure

This repository is best understood as a collection of related experiments around the same research theme rather than a single polished training package.

- [mnist_test.py](/Users/mamgo/Desktop/CNN-with-PDE/mnist_test.py): grayscale PDE diffusion classifier for MNIST with Strang splitting, smoothed coefficients, and batched Thomas solver.
- [fashion_mnist.py](/Users/mamgo/Desktop/CNN-with-PDE/fashion_mnist.py): Fashion-MNIST version with a larger MLP head and batch normalization.
- [SVHN.py](/Users/mamgo/Desktop/CNN-with-PDE/SVHN.py): RGB diffusion model with channel coupling and skip connection to preserve image detail.
- [cifar10.py](/Users/mamgo/Desktop/CNN-with-PDE/cifar10.py): more experimental CIFAR-10 architecture with multiple PDE scales, channel mixing, and attention-like components.
- [cifar_2version.py](/Users/mamgo/Desktop/CNN-with-PDE/cifar_2version.py): hybrid PDE experiment combining diffusion-inspired and other PDE-motivated blocks.
- [emotion_recognition.py](/Users/mamgo/Desktop/CNN-with-PDE/emotion_recognition.py): facial emotion classification using a PDE preprocessing layer and an MLP classifier.
- [tiny_imagenet.py](/Users/mamgo/Desktop/CNN-with-PDE/tiny_imagenet.py): larger-scale experiment on TinyImageNet with a more simplified diffusion formulation.

## Main Technical Contributions

If you are discussing this project in an interview, the most defensible contributions are:

- **Learnable diffusion coefficients:** the model learns $\alpha$ and $\beta$ instead of fixed smoothing strengths.
- **Time-dependent PDE parameters:** several experiments let the diffusion coefficients evolve over artificial time.
- **Differentiable numerical solver inside the network:** training happens through finite-difference updates and batched tridiagonal solves.
- **Operator-splitting-based layer design:** the 2D PDE is made practical through sequential 1D implicit solves.
- **Exploration across data regimes:** the method is tested on simple grayscale digits, fashion items, RGB house numbers, and more challenging natural-image datasets.

## What the Code Actually Shows Empirically

From the current repository and script comments/output strings:

- **MNIST:** about `97.33%` test accuracy is explicitly recorded in [mnist_test.py](/Users/mamgo/Desktop/CNN-with-PDE/mnist_test.py:1).
- **Fashion-MNIST:** the script reports overall test accuracy at evaluation time, but no fixed benchmark number is stored in the source header.
- **SVHN:** the script tracks validation and final test accuracy, again without a single canonical number fixed in the README-worthy source comments.
- **CIFAR-10 / TinyImageNet:** these experiments are more exploratory and demonstrate where pure diffusion-based feature extraction starts to struggle on complex natural images.

The broad pattern is scientifically plausible:

- PDE diffusion works relatively well when the dataset has strong low-level structure and clean shapes.
- Performance degrades as texture complexity, semantic variability, and scale diversity increase.
- This suggests diffusion alone is a useful inductive bias, but not a full replacement for the representation power of modern convolutional or attention-based models on hard natural-image tasks.

## Research Interpretation

One useful way to frame the project is:

> This repository is a study of PDE-based inductive biases for vision, not an attempt to beat state-of-the-art CNNs or ViTs.

That framing is strong in interviews because it is honest and intellectually mature.

The project demonstrates:

- how continuous operators can be translated into trainable neural layers,
- how stability constraints shape architecture design,
- how numerical analysis decisions affect learning behavior,
- how mathematically structured models can improve interpretability.

It also reveals real limitations:

- pure diffusion is inherently smoothing, so discriminative high-frequency detail can be lost,
- natural images often require richer hierarchical feature extraction,
- parameterizing only diffusion may be too restrictive for complex semantics,
- some experiment scripts are prototype-quality research code rather than a unified benchmark framework.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment directly, for example:

```bash
python mnist_test.py
python fashion_mnist.py
python SVHN.py
python cifar10.py
```

Notes:

- several scripts download datasets automatically through `torchvision`,
- the scripts are standalone and do not share a single training CLI,
- some scripts include visualization code for learned coefficient fields and diffused outputs.

## Mathematical Discussion Points for Interviews

If you present this project to an AI research engineer interviewer, these are the strongest talking points:

### 1. PDE View of Representation Learning

A residual network can be interpreted as a discretized dynamical system. This project pushes that idea further by choosing a specific continuous operator, diffusion, as the feature evolution mechanism.

### 2. Stability Matters

In learned dynamical systems, unstable updates can destroy both optimization and representation quality. Here, positivity constraints on $\alpha,\beta$, coefficient smoothing, implicit solves, and conservative time steps are all used to keep the PDE layer numerically well-behaved.

### 3. Inductive Bias Tradeoff

Diffusion encodes a strong prior: nearby pixels should interact smoothly. That prior helps on simpler datasets but can oversmooth discriminative structure on more complex images. This is a useful example of the bias-variance tradeoff in architecture design.

### 4. Scientific Computing Meets Deep Learning

This repository is a concrete example of integrating:

- finite differences,
- operator splitting,
- tridiagonal linear solvers,
- learnable coefficient fields,
- end-to-end gradient-based optimization.

### 5. Honest Next Steps

The natural next research directions are:

- adding reaction or transport terms, not only diffusion,
- learning coefficients with small neural networks rather than raw parameter grids,
- extending to nonlinear diffusion,
- coupling PDE layers with CNN or transformer backbones,
- performing ablations on coefficient parameterization, boundary conditions, and solver choice,
- benchmarking with standardized experiment tracking and reproducibility controls.
