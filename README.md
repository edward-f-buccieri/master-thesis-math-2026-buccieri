# Gradient Flows and Optimal Transport in Neural Network Training

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository containing the source code and numerical experiments for my Master's Thesis in Mathematics (University of Pavia, 2026).

## 🎥 Simulations

| Standard Approximation |
|:----------------------:|
| <img src="Animations/jko_evolution_abs.gif" width="800"/> |
| *Evolution of the Wasserstein gradient flow for a ReLU network approximating the target function* $y(x)=\vert x \vert$ |
| <img src="Animations/jko_evolution_quad.gif" width="800"/> |
| *Evolution of the Wasserstein gradient flow for a ReLU network approximating the target function* $y(x)=x^2$ |

| Approximation with noisy data |
|:-----------------------------:|
| <img src="Animations/jko_evolution_noisy.gif" width="800"/> |
| *Evolution of the Wasserstein gradient flow for a ReLU network approximating a noisy target function* $y(x)=x^2 + 0.05 \cdot Z$ *, where* $Z\sim\mathcal{N}(0,1)$ |

| Transport dynamics with two neurons |
|:-----------------------------------:|
| <img src="Animations/jko_evolution_two_neurons.gif" width="800"/> |
| *Evolution of a ReLU network with two neurons and of its parameters approximating the target function* $y(x)=\max\{x-1,0\}$ |

## 🧠 Abstract
The goal of my Master's thesis is to describe the training of shallow neural networks as a Wasserstein gradient flow. In the experiments, I chose different target functions to represent the dataset. Starting from an initial parameter distribution for the network, I simulated the time evolution of this distribution during training (i.e., the approximation of the target function) using the JKO scheme. This repository provides some simple examples of target functions, but the idea can be easily extended to more complex scenarios.

## 📂 Repository Structure
```text
master-thesis-math-2026-buccieri/
│
├── Animations/               # Script generating the animations seen in this page
├── experiments/              # Scripts generating the figures included in my thesis
├── requirements.txt          # Project dependencies
└── README.md
```


## ⚙️ Requirements & Hardware

The numerical simulations rely heavily on optimal transport solvers which are optimized for GPU execution. The original experiments were conducted on **Google Colab** utilizing **NVIDIA T4** and **NVIDIA A100** GPUs. Running the JKO scheme simulations on a standard CPU is possible but will result in significantly longer execution times.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/edward-f-buccieri/master-thesis-math-2026-buccieri.git
cd master-thesis-math-2026-buccieri
pip install -r requirements.txt
