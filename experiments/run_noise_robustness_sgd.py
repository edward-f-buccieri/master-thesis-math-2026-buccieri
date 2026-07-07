import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# Quadratic target function
def target_func(x):
    return x**2


# ReLU network
def network_forward(x, mu):
    w = mu[:, 0:1].t()
    theta = mu[:, 1:2].t()
    b = mu[:, 2:3].t()
    return (w * torch.relu(x @ theta + b)).mean(dim=1, keepdim=True)


# Hyperparameters
num_particles = 2000  # Total number of neurons / particles
iterations = 150000     # Total number of SGD steps
batch_size = 200       # Mini-batch size for SGD
lambda_v = 1e-5       # L2 regularization coefficient
lr = 2              # Learning rate


# Training set
noise_std = 0.05
x_train = torch.linspace(-1, 1, 200).view(-1, 1).to(device)
y_train = target_func(x_train) + torch.randn_like(x_train) * noise_std


# Particle initialization on the sphere in R^3
R = 1.0
directions = torch.randn(num_particles, 3, device=device)
directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
mu_0 = R * directions


# Loss function
criterion = nn.MSELoss()


# Setup SGD Optimizer directly on the particles
current_mu = mu_0.clone().requires_grad_(True)
optimizer = optim.SGD([current_mu], lr=lr)


# Training history
history_mu = {}
history_out = {}
step_keys = [0, iterations // 2, iterations]


# Save initial state
with torch.no_grad():
    history_mu[0] = current_mu.clone().detach().cpu().numpy()
    history_out[0] = network_forward(x_train, current_mu).squeeze(1).cpu().numpy()


# SGD loop
for step in range(1, iterations + 1):
    optimizer.zero_grad(set_to_none=True)

    # Mini-batch sampling (Stochasticity)
    indices = torch.randperm(x_train.size(0))[:batch_size]
    x_batch = x_train[indices]
    y_batch = y_train[indices]

    # Network forward pass on the mini-batch
    output = network_forward(x_batch, current_mu)

    # Energy functional: data loss plus L2 regularization (2-homogeneous)
    loss_F = criterion(output, y_batch) + (lambda_v / 2.0) * torch.mean(current_mu**2)

    # SGD step
    loss_F.backward()
    optimizer.step()

    # Save selected intermediate states
    if step in step_keys:
        print(f"SGD step {step}/{iterations} completed. Loss: {loss_F.item():.4f}")
        with torch.no_grad():
            history_mu[step] = current_mu.clone().detach().cpu().numpy()
            history_out[step] = network_forward(x_train, current_mu).squeeze(1).cpu().numpy()


# Final empirical error evaluated on the full dataset
with torch.no_grad():
    final_output = network_forward(x_train, current_mu)
    final_mse = criterion(final_output, y_train).item()

print(f"Final MSE: {final_mse:.6e}")


# Visualization
x_np = x_train.cpu().numpy().flatten()
y_np = y_train.cpu().numpy().flatten()
titles = ["Initialization", f"{iterations // 2} SGD steps", f"{iterations} SGD steps"]
fig = plt.figure(figsize=(20, 11), constrained_layout=True)
fig.suptitle(r"ReLU network - Target: $y(x) = x^2$ (Particle Gradient Flow)", fontsize=16)

for idx, step in enumerate(step_keys):
    # Function approximation
    ax_func = fig.add_subplot(2, 3, idx + 1)
    ax_func.scatter(x_np, y_np, color='gray', s=10, alpha=0.4, label="Target")
    ax_func.plot(x_np, history_out[step], "b-", label=r"Network $\Phi_\mu$", linewidth=2)
    ax_func.set_title(titles[idx])
    ax_func.grid(True, alpha=0.3)
    ax_func.legend(loc="lower left")
    ax_func.set_ylim([-0.2, 1.2])

    # Particle distribution in 3D: (theta, b, w)
    ax_3d = fig.add_subplot(2, 3, idx + 4, projection="3d")
    pts = history_mu[step]

    # Color points by the output weight w
    sc = ax_3d.scatter(
        pts[:, 1],
        pts[:, 2],
        pts[:, 0],
        c=pts[:, 0],
        cmap="viridis",
        s=10,
        alpha=0.6,
    )

    ax_3d.set_xlabel(r"$\theta$", labelpad=8)
    ax_3d.set_ylabel(r"$b$", labelpad=8)
    ax_3d.set_zlabel(r"$w$", labelpad=8)
    ax_3d.set_xlim([-3, 3])
    ax_3d.set_ylim([-3, 3])
    ax_3d.set_zlim([-3, 3])
    ax_3d.view_init(elev=25, azim=-60)

    fig.colorbar(sc, ax=ax_3d, shrink=0.75, pad=0.08, label="w")

# Save and show the figures
plt.savefig("particle_gradient_flow_noisy.png", dpi=300, bbox_inches="tight", pad_inches=0.35)
plt.show()
