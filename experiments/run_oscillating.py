import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from geomloss import SamplesLoss

# Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Oscillating target function
def target_func(x):
    signal = torch.sin(2 * torch.pi * x) + 0.5 * torch.cos(6 * torch.pi * x)
    return signal / 1.5

# ReLU network
def network_forward(x, mu):
    w = mu[:, 0:1].t()
    theta = mu[:, 1:2].t()
    b = mu[:, 2:3].t()
    return (w * torch.relu(x @ theta + b)).mean(dim=1, keepdim=True)

# Hyperparameters
num_particles = 4000  # Total number of neurons / particles
jko_steps = 300        # Number of JKO iterations
inner_iters = 120      # Number of optimizer iterations inside each JKO step
tau = 0.8          # JKO time step
blur_eps = 0.005       # Sinkhorn blur parameter
lambda_v = 1e-3       # L2 regularization coefficient
lr_inner = 0.02        # Inner optimizer learning rate

# Training set
x_train = torch.linspace(-1, 1, 400).view(-1, 1).to(device)
y_train = target_func(x_train)

# Particle initialization on the sphere in R^3
R = 1.0
directions = torch.randn(num_particles, 3, device=device)
directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
mu_0 = R * directions


# Loss functions
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur_eps)
criterion = nn.MSELoss()


# Training history
history_mu = {}
history_out = {}
step_keys = [0, jko_steps // 2, jko_steps]

current_mu = mu_0.clone()

# Save initial state
with torch.no_grad():
    history_mu[0] = current_mu.cpu().numpy()
    history_out[0] = network_forward(x_train, current_mu).squeeze(1).cpu().numpy()

# JKO iterations
for step in range(1, jko_steps + 1):
    previous_mu = current_mu.detach()
    mu_k = previous_mu.clone().requires_grad_(True)
    optimizer = optim.Adam([mu_k], lr=lr_inner)

    for _ in range(inner_iters):
        optimizer.zero_grad(set_to_none=True)

        # Network forward pass
        output = network_forward(x_train, mu_k)

        # Energy functional: data loss plus L2 regularization (2-homogeneous)
        loss_F = criterion(output, y_train) + (lambda_v / 2.0) * torch.mean(mu_k**2)

        # Sinkhorn approximation of the Wasserstein-2 transport term
        loss_W2 = sinkhorn_loss(mu_k, previous_mu)

        # JKO objective
        jko_obj = loss_F + (1.0 / (2.0 * tau)) * loss_W2
        jko_obj.backward()
        optimizer.step()

    current_mu = mu_k.detach()

    if step % 25 == 0:
        with torch.no_grad():
          mse = criterion(network_forward(x_train, current_mu), y_train).item()
        print(f"Step {step}/{jko_steps} | MSE: {mse:.6e}")

    # Save selected intermediate states
    if step in step_keys:
        print(f"JKO step {step}/{jko_steps} completed. JKO objective: {jko_obj.item():.4f}")
        with torch.no_grad():
            history_mu[step] = current_mu.cpu().numpy()
            history_out[step] = network_forward(x_train, current_mu).squeeze(1).cpu().numpy()

# Final empirical error
with torch.no_grad():
    final_output = network_forward(x_train, current_mu)
    final_mse = criterion(final_output, y_train).item()

print(f"Final MSE: {final_mse:.6e}")

# Visualization
x_np = x_train.cpu().numpy().flatten()
y_np = y_train.cpu().numpy().flatten()
titles = ["Initialization", f"{jko_steps // 2} JKO steps", f"{jko_steps} JKO steps"]
fig = plt.figure(figsize=(20, 11))
fig.suptitle(r"ReLU network - Target: Oscillating Function", fontsize=16, y=0.95)

# Pre-calculate global limits across all saved timesteps
global_lim = max([np.max(np.abs(pts)) for pts in history_mu.values()])
lim = global_lim * 1.05  # 5% padding for rendering borders safely

global_w_min = min([np.min(pts[:, 0]) for pts in history_mu.values()])
global_w_max = max([np.max(pts[:, 0]) for pts in history_mu.values()])

for idx, step in enumerate(step_keys):
    # Function approximation
    ax_func = fig.add_subplot(2, 3, idx + 1)
    ax_func.plot(x_np, y_np, "k--", label="Target", linewidth=2)
    ax_func.plot(x_np, history_out[step], "g-", label=r"Network $\Phi_\mu$", linewidth=2)
    ax_func.set_title(titles[idx], fontsize=12, pad=10)
    ax_func.grid(True, alpha=0.3)
    ax_func.legend(loc="lower left")
    ax_func.set_ylim([-1.7, 1.7])

    # Particle distribution in 3D: (theta, b, w)
    ax_3d = fig.add_subplot(2, 3, idx + 4, projection="3d")
    pts = history_mu[step]

    sc = ax_3d.scatter(
        pts[:, 1], 
        pts[:, 2], 
        pts[:, 0],
        c=pts[:, 0],
        cmap="magma",
        s=10,
        alpha=0.6,
        vmin=global_w_min,
        vmax=global_w_max
    )

    ax_3d.set_xlabel(r"$\theta$", labelpad=8)
    ax_3d.set_ylabel(r"$b$", labelpad=8)
    ax_3d.set_zlabel(r"$w$", labelpad=8)

    ax_3d.set_xlim([-lim, lim])
    ax_3d.set_ylim([-lim, lim])
    ax_3d.set_zlim([-lim, lim])
    ax_3d.view_init(elev=25, azim=-60)

    fig.colorbar(sc, ax=ax_3d, shrink=0.7, pad=0.05, label="w")

    # Focus on the initial sphere (upper-left area of the first 3D plot)
    if idx == 0:
        ax_inset = fig.add_axes([0.04, 0.32, 0.09, 0.14], projection='3d')

        ax_inset.scatter(
            pts[:, 1],
            pts[:, 2],
            pts[:, 0],
            c=pts[:, 0],
            cmap="magma",
            s=4,        
            alpha=0.5,
            vmin=global_w_min,
            vmax=global_w_max
        )

        inset_lim = 1.2
        ax_inset.set_xlim([-inset_lim, inset_lim])
        ax_inset.set_ylim([-inset_lim, inset_lim])
        ax_inset.set_zlim([-inset_lim, inset_lim])
        ax_inset.view_init(elev=25, azim=-60)

        ax_inset.set_title("Zoom (R=1)", fontsize=9, pad=2, weight='bold')
        ax_inset.set_xticks([-1, 0, 1])
        ax_inset.set_yticks([-1, 0, 1])
        ax_inset.set_zticks([-1, 0, 1])
        ax_inset.tick_params(labelsize=7, pad=0)
        
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.88, wspace=0.25, hspace=0.28)

# Save and show the figures
plt.savefig("jko_sinkhorn_osc_fixed.png", dpi=300, bbox_inches="tight", pad_inches=0.35)
plt.show()
