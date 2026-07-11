import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.optim as optim
from geomloss import SamplesLoss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# Target Function: ReLU(x - 1)
def target_func(x):
    return torch.relu(x - 1.0)

# ReLU network
def network_forward(x, mu):
    w = mu[:, 0:1].t()
    theta = mu[:, 1:2].t()
    b = mu[:, 2:3].t()
    activations = w * torch.relu(x @ theta + b)
    return activations.mean(dim=1, keepdim=True), activations / mu.shape[0]

# Hyperparameters
num_particles = 2   # Number of neurons
jko_steps = 48   # Number of JKO iterations
inner_iters = 30   # Number of optimizer iterations inside each JKO step
tau = 0.5           # JKO time step
eps = 0.006     # Sinkhorn regularization parameter
lambda_v = 5*1e-3     # L^2 Regularization coefficient
lr_inner = 0.05     # Inner optimizer learning rate

# Training set
x_train = torch.linspace(-10, 10, 200).view(-1, 1).to(device)
y_train = target_func(x_train).to(device)

# Evaluation set
x_eval = torch.linspace(-40, 40, 400).view(-1, 1).to(device)
y_eval = target_func(x_eval).to(device)

# Particle initialization
mu_0 = torch.zeros([num_particles, 3], dtype=torch.float32, device=device)
mu_0[0,:]=torch.tensor([1.2, -1.0, -1.5]).to(device)
mu_0[1,:]=torch.tensor([-0.8, 1.0, -0.5]).to(device)

# Loss functions
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=eps)
criterion = nn.MSELoss()

# Training history
history_mu = {}
history_out = {}
history_neurons = {}
step_keys = [0, jko_steps // 5, jko_steps * 2 // 5, jko_steps * 3 // 5, jko_steps * 4 // 5, jko_steps]

current_mu = mu_0.clone()

# Save initial state
with torch.no_grad():
    history_mu[0] = current_mu.cpu().numpy()
    history_out[0] = network_forward(x_eval, current_mu)[0].squeeze(1).cpu().numpy()
    history_neurons[0] = network_forward(x_eval, current_mu)[1].cpu().numpy()

for step in range(1, jko_steps + 1):
    previous_mu = current_mu.detach()
    mu_k = previous_mu.clone().requires_grad_(True)
    optimizer = optim.Adam([mu_k], lr=lr_inner)

    for _ in range(inner_iters):
        optimizer.zero_grad(set_to_none=True)

        # Network forward pass
        output = network_forward(x_train, mu_k)[0]

        # Energy functional: data loss plus L2 regularization (2-homogeneous)
        loss_F = criterion(output, y_train) + (lambda_v / 2.0) * torch.mean(mu_k**2)

        # Sinkhorn approximation of the Wasserstein-2 transport term
        loss_W2 = sinkhorn_loss(mu_k, previous_mu)

        # JKO objective
        jko_obj = loss_F + (1.0 / (2.0 * tau)) * loss_W2
        jko_obj.backward()
        optimizer.step()

    current_mu = mu_k.detach()

    # Save selected intermediate states
    if step in step_keys:
        print(f"JKO step {step}/{jko_steps} completed. JKO objective: {jko_obj.item():.4f}")
        with torch.no_grad():
            history_mu[step] = current_mu.cpu().numpy()
            history_out[step] = network_forward(x_eval, current_mu)[0].squeeze(1).cpu().numpy()
            history_neurons[step] = network_forward(x_eval, current_mu)[1].cpu().numpy()

# Final empirical error
with torch.no_grad():
    final_output = network_forward(x_train, current_mu)[0]
    final_mse = criterion(final_output, y_train).item()

print(f"Final MSE: {final_mse:.6e}")

# Visualization
x_np = x_eval.cpu().numpy().flatten()
y_np = y_eval.cpu().numpy().flatten()
titles = ["Initialization", f"{jko_steps // 5} JKO steps", f"{jko_steps * 2 // 5} JKO steps", f"{jko_steps * 3 // 5} JKO steps", f"{jko_steps * 4 // 5} JKO steps", f"{jko_steps} JKO steps"]
fig = plt.figure(figsize=(20, 11), constrained_layout=True)
fig.suptitle(r"ReLU network - Target: ReLU$(x-1)$", fontsize=16)
fig_2 = plt.figure(figsize=(20, 11), constrained_layout=True)
fig_2.suptitle(r"ReLU network - Target: ReLU$(x-1)$", fontsize=16)
cmap = cm.get_cmap('tab10') # Generate colors for the neurons
y_min_global = min(y_np.min(), min(history_out[k].min() for k in step_keys))
y_max_global = max(y_np.max(), max(history_out[k].max() for k in step_keys))
padding = (y_max_global - y_min_global) * 0.1

for idx, step in enumerate(step_keys):
    # Function approximation
    ax_func = fig.add_subplot(2, 3, idx + 1)
    ax_func.plot(x_np, y_np, "k--", label="Target", linewidth=2)

    # Show all the neurons
    for i in range(num_particles):
        color = cmap(i)
        ax_func.plot(x_np, history_neurons[step][:,i], color=color, linestyle='-', label=f"Neuron {i+1}", linewidth=2)

    ax_func.plot(x_np, history_out[step], "g-", label=r"Network $\Phi_\mu$", linewidth=4, alpha=0.6)

    ax_func.set_title(titles[idx])
    ax_func.grid(True, alpha=0.3)
    ax_func.legend(loc="lower left")
    ax_func.set_ylim([y_min_global - padding, y_max_global + padding])

    # Particle distribution in 3D: (theta, b, w)
    ax_3d = fig_2.add_subplot(2, 3, idx + 1, projection="3d")
    pts = history_mu[step]

    for i in range(num_particles):
        color = cmap(i)
        w_val = pts[i, 0]
        theta_val = pts[i, 1]
        b_val = pts[i, 2]

        ax_3d.scatter(
            theta_val, b_val, w_val,
            color=color,
            s=80,
            alpha=0.9,
            label=f"N{i+1}: w={w_val:+.2f}"
        )

    ax_3d.set_title(titles[idx])
    ax_3d.set_xlabel(r"$\theta$", labelpad=8)
    ax_3d.set_ylabel(r"$b$", labelpad=8)
    ax_3d.set_zlabel(r"$w$", labelpad=8)
    ax_3d.set_xlim([-3, 3])
    ax_3d.set_ylim([-3, 3])
    ax_3d.set_zlim([-3, 3])
    ax_3d.view_init(elev=25, azim=-60)

    ax_3d.legend(loc="upper left", fontsize=10)

# Save and show the figures
fig.savefig("dynamics_two_neurons_approx.png", dpi=300, bbox_inches="tight", pad_inches=0.35)
fig_2.savefig("dynamics_two_neurons.png", dpi=300, bbox_inches="tight", pad_inches=0.35)
plt.show()
