import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.optim as optim
from geomloss import SamplesLoss
from pathlib import Path

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
num_particles = 2 # Total number of neurons / particles
jko_steps = 24    # Number of JKO iterations
inner_iters = 30  # Number of optimizer iterations inside each JKO step
tau = 0.5         # JKO time step
blur_eps = 0.006  # Sinkhorn blur parameter
lambda_v = 5e-3   # L2 regularization coefficient
lr_inner = 0.05   # Inner optimizer learning rate

# Training Set
x_train = torch.linspace(-10, 10, 200).view(-1, 1).to(device)
y_train = target_func(x_train).to(device)

# Particle initialization
mu_0 = torch.zeros([num_particles, 3], dtype=torch.float32, device=device)
mu_0[0, :] = torch.tensor([2.2, -1.0, -1.5], device=device)
mu_0[1, :] = torch.tensor([-0.8, 1.0, -0.5], device=device)

# Loss functions
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur_eps)
criterion = nn.MSELoss()

# Training history (\mu, \Phi_\mu and single neurons)
history_mu = []
history_out = []
history_neurons = []

current_mu = mu_0.clone()

# Save initial state (\mu_0, \Phi_{\mu_0} and single neurons)
with torch.no_grad():
    net_out, neuron_out = network_forward(x_train, current_mu)
    history_mu.append(current_mu.cpu().numpy())
    history_out.append(net_out.squeeze(1).cpu().numpy())
    history_neurons.append(neuron_out.cpu().numpy())

# JKO iterations
print("Starting optimization with JKO scheme...")
for step in range(1, jko_steps + 1):
    mu_k = current_mu.clone().requires_grad_(True)
    optimizer = optim.Adam([mu_k], lr=lr_inner)

    for _ in range(inner_iters):
        optimizer.zero_grad(set_to_none=True)

        output = network_forward(x_train, mu_k)[0]
        loss_F = criterion(output, y_train) + (lambda_v / 2.0) * torch.mean(mu_k**2)
        loss_W2 = sinkhorn_loss(mu_k, current_mu)
        jko_obj = loss_F + (1.0 / (2 * tau)) * loss_W2

        jko_obj.backward()
        optimizer.step()

    current_mu = mu_k.detach()

    print(f"Step {step}/{jko_steps} processed. JKO Obj: {jko_obj.item():.4f}")
    with torch.no_grad():
        net_out, neuron_out = network_forward(x_train, current_mu)
        history_mu.append(current_mu.cpu().numpy())
        history_out.append(net_out.squeeze(1).cpu().numpy())
        history_neurons.append(neuron_out.cpu().numpy())

print("Optimization completed. Generating animation...")

# Convert histories to arrays for easier plotting
history_mu = np.asarray(history_mu)  
history_out = np.asarray(history_out)
history_neurons = np.asarray(history_neurons)

x_np = x_train.cpu().numpy().flatten()
y_np = y_train.cpu().numpy().flatten()
frames_count = len(history_mu)

# Animation configuration
fig = plt.figure(figsize=(14, 6), dpi=150)
fig.suptitle(r"ReLU network - Target: ReLU(x - 1)", fontsize=16)

neuron_cmap = cm.get_cmap("tab10")
neuron_colors = [neuron_cmap(i % 10) for i in range(num_particles)]
linestyles = ['--', ':', '-.', '-']

# Function approximation and single neurons
ax_2d = fig.add_subplot(1, 2, 1)
ax_2d.plot(x_np, y_np, "k--", label="Target", linewidth=2)

neuron_lines = []
for i in range(num_particles):
    line_neuron, = ax_2d.plot(
        x_np,
        history_neurons[0, :, i],
        color=neuron_colors[i],
        linestyle=linestyles[i % len(linestyles)],
        linewidth=1.8,
        alpha=0.85,
        label=f"Neuron {i + 1}",
    )
    neuron_lines.append(line_neuron)

line_net, = ax_2d.plot(
    x_np,
    history_out[0],
    color="green",
    linestyle="-",
    label=r"Network $\Phi_\mu$",
    linewidth=2.5,
)

y_all = np.concatenate(
    [
        y_np.ravel(),
        history_out.ravel(),
        history_neurons.reshape(-1),
    ]
)
y_pad = 0.08 * (y_all.max() - y_all.min() + 1e-8)
ax_2d.set_xlim(float(x_np.min()), float(x_np.max()))
ax_2d.set_ylim(float(y_all.min() - y_pad), float(y_all.max() + y_pad))
ax_2d.set_xlabel("x")
ax_2d.set_ylabel("output")
ax_2d.grid(True, alpha=0.3)
ax_2d.legend(loc="upper left", fontsize=8)

# Support evolution in 3D
ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

theta_vals = history_mu[:, :, 1].ravel()
b_vals = history_mu[:, :, 2].ravel()
w_vals = history_mu[:, :, 0].ravel()

def padded_limits(values, pad_ratio=0.12):
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    pad = pad_ratio * (v_max - v_min + 1e-8)
    return v_min - pad, v_max + pad

theta_lim = padded_limits(theta_vals)
b_lim = padded_limits(b_vals)
w_lim = padded_limits(w_vals)

def setup_3d_axis():
    ax_3d.set_xlabel(r"$\theta$")
    ax_3d.set_ylabel(r"$b$")
    ax_3d.set_zlabel(r"$w$")
    ax_3d.set_xlim(theta_lim)
    ax_3d.set_ylim(b_lim)
    ax_3d.set_zlim(w_lim)
    ax_3d.view_init(elev=24, azim=-55)
    ax_3d.grid(True, alpha=0.25)

def update(frame):
    line_net.set_ydata(history_out[frame])
    for i, line_neuron in enumerate(neuron_lines):
        line_neuron.set_ydata(history_neurons[frame, :, i])

    ax_2d.set_title(f"Function approximation (JKO step {frame}/{jko_steps})")

    ax_3d.clear()
    setup_3d_axis()
    ax_3d.set_title("Neuron paths in parameter space")

    for i in range(num_particles):
        path = history_mu[: frame + 1, i, :]
        current = history_mu[frame, i, :]
        color = neuron_colors[i]

        ax_3d.plot(
            path[:, 1],
            path[:, 2],
            path[:, 0],
            color=color,
            linewidth=2,
            alpha=0.9,
        )
        ax_3d.scatter(
            current[1],
            current[2],
            current[0],
            color=[color],
            s=55,
            edgecolor="black",
            linewidth=0.5,
            label=f"Neuron {i + 1}",
        )

    ax_3d.legend(loc="upper left", fontsize=8)
    return [line_net, *neuron_lines]


ani = animation.FuncAnimation(
    fig,
    update,
    frames=frames_count,
    interval=120,
    blit=False,
)

plt.tight_layout()

print("Saving the animation...")
ani.save("jko_evolution_two_neurons.gif", writer="pillow", fps=1, dpi=150)
print(f"File jko_evolution_two_neurons.gif saved with success!")
