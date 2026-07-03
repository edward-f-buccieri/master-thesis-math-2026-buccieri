import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

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

# Training Set
x_train = torch.linspace(-1, 1, 400).view(-1, 1).to(device)
y_train = target_func(x_train).to(device)

# Particle intialization on the sphere in 3D
R = 1.0
directions = torch.randn(num_particles, 3, device=device)
directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
mu_0 = directions * R

# Loss functions
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur_eps)
criterion = nn.MSELoss()

# Training history (\mu and \Phi_\mu)
history_mu = []
history_out = [] 

current_mu = mu_0.clone()

# Save initial state (\mu_0 and \Phi_{\mu_0})
with torch.no_grad():
    history_mu.append(current_mu.cpu().numpy())
    history_out.append(network_forward(x_train, current_mu).squeeze(1).cpu().numpy())

# JKO iterations
print("Starting optimization with JKO scheme...")
for step in range(1, jko_steps + 1):
    mu_k = current_mu.clone().requires_grad_(True)
    optimizer = optim.Adam([mu_k], lr=lr_inner)

    for i in range(inner_iters):
        optimizer.zero_grad(set_to_none=True)
        
        # Network forward pass
        output = network_forward(x_train, mu_k)

        # Energy functional: data loss plus L2 regularization (2-homogeneous)
        loss_F = criterion(output, y_train) + (lambda_v / 2.0) * torch.mean(mu_k**2)

        # Sinkhorn approximation of the Wasserstein-2 transport term
        loss_W2 = sinkhorn_loss(mu_k, current_mu)

        # JKO objective
        jko_obj = loss_F + (1.0 / (2 * tau)) * loss_W2
        jko_obj.backward()
        optimizer.step()

    current_mu = mu_k.detach()

    # Save all the JKO step
    print(f"Step {step}/{jko_steps} processed. JKO Obj: {jko_obj.item():.4f}")
    with torch.no_grad():
        history_mu.append(current_mu.cpu().numpy())
        history_out.append(network_forward(x_train, current_mu).squeeze(1).cpu().numpy())  

print("Optimization completed. Generating animations...")

# Animation configuration
fig = plt.figure(figsize=(14, 6), dpi=150)
fig.suptitle(r"ReLU network - Target : Oscillating Function", fontsize=16)

# Pre-calculate global limits across all saved timesteps
global_lim = max([np.max(np.abs(pts)) for pts in history_mu.values()])
lim = global_lim * 1.05  # 5% padding for rendering borders safely

global_w_min = min([np.min(pts[:, 0]) for pts in history_mu.values()])
global_w_max = max([np.max(pts[:, 0]) for pts in history_mu.values()])

# Function approximation
ax_2d = fig.add_subplot(1, 2, 1)
x_np = x_train.cpu().numpy().flatten()
y_np = y_train.cpu().numpy().flatten()

ax_2d.plot(x_np, y_np, 'k--', label='Target', linewidth=2)
line_net, = ax_2d.plot(x_np, history_out[0], 'g-', label=r'Network $\Phi_\mu$', linewidth=2)
ax_2d.set_ylim([-1.7, 1.7])
ax_2d.grid(True, alpha=0.3)
ax_2d.legend(loc='upper right')

# Support evolution in 3D
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
cmap = cm.get_cmap('magma')
norm = plt.Normalize(vmin=-1.0, vmax=1.5)

def update(frame):
    line_net.set_ydata(history_out[frame])
    ax_2d.set_title(f"Function Approximation (Step {frame}/{jko_steps})")

    ax_3d.clear()
    pts = history_mu[frame]

    w_vals = pts[:, 0]
    w_abs = np.abs(w_vals)
    w_max = np.max(w_abs) + 1e-8

    colors = cmap(norm(w_vals))
    colors[:, 3] = np.clip(w_abs / w_max, 0.05, 1.0)

    ax_3d.scatter(pts[:, 1], pts[:, 2], pts[:, 0], c=colors, s=10)

    ax_3d.set_xlabel(r'$\theta$')
    ax_3d.set_ylabel(r'$b$')
    ax_3d.set_zlabel(r'$w$')

    ax_3d.set_xlim([-lim, lim])
    ax_3d.set_ylim([-lim, lim])
    ax_3d.set_zlim([-lim, lim])
    ax_3d.set_title("Support Evolution")

    return line_net,

ani = animation.FuncAnimation(fig, update, frames=len(history_mu), interval=60, blit=False)

print("Saving the animations...")

# Save the animation in GIF format (DPI=150, FPS=15)
ani.save("jko_evolution_osc.gif", writer='pillow', fps=15, dpi=150)
print("File jko_evolution_osc.gif saved with success!")
