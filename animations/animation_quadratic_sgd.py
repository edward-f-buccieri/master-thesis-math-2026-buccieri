import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
num_particles = 2000    # Total number of neurons / particles
iterations = 180000     # Total number of SGD steps
batch_size = 200         # Mini-batch size for SGD
lambda_v = 1e-5         # L2 regularization coefficient
lr = 2.0                # Learning rate
save_interval = iterations // 100  # Save history periodically for the animation (100 frames)

# Training Set
x_train = torch.linspace(-1, 1, 200).view(-1, 1).to(device)
y_train = target_func(x_train).to(device)

# Particle intialization on the sphere in 3D
R = 1.0
directions = torch.randn(num_particles, 3, device=device)
directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
mu_0 = directions * R

# Loss function
criterion = nn.MSELoss()

# Setup SGD Optimizer directly on the particles
current_mu = mu_0.clone().requires_grad_(True)
optimizer = optim.SGD([current_mu], lr=lr)

# Training history (\mu, \Phi_\mu and step number)
history_mu = []
history_out = [] 
history_steps = []

# Save initial state (\mu_0 and \Phi_{\mu_0})
with torch.no_grad():
    history_mu.append(current_mu.clone().detach().cpu().numpy())
    history_out.append(network_forward(x_train, current_mu).squeeze(1).cpu().numpy())
    history_steps.append(0)

# SGD loop
print("Starting optimization with SGD scheme...")
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

    # Save selected intermediate states periodically for the animation
    if step % save_interval == 0 or step == iterations:
        print(f"Step {step}/{iterations} processed. Loss: {loss_F.item():.6f}")
        with torch.no_grad():
            history_mu.append(current_mu.clone().detach().cpu().numpy())
            history_out.append(network_forward(x_train, current_mu).squeeze(1).cpu().numpy())
            history_steps.append(step)

print("Optimization completed. Generating animations...")

# Animation configuration
fig = plt.figure(figsize=(14, 6), dpi=150)
fig.suptitle(r"ReLU network - Target: $y(x) = x^2$ (Particle Gradient Flow)", fontsize=16)

# Function approximation
ax_2d = fig.add_subplot(1, 2, 1)
x_np = x_train.cpu().numpy().flatten()
y_np = y_train.cpu().numpy().flatten()

ax_2d.plot(x_np, y_np, 'k--', label='Target', linewidth=2)
line_net, = ax_2d.plot(x_np, history_out[0], 'b-', label=r'Network $\Phi_\mu$', linewidth=2)
ax_2d.set_ylim([-0.2, 1.2])
ax_2d.grid(True, alpha=0.3)
ax_2d.legend(loc='upper center')

# Support evolution in 3D
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
cmap = cm.get_cmap('viridis')
norm = plt.Normalize(vmin=-1.0, vmax=4.0)

def update(frame):
    current_step = history_steps[frame]
    line_net.set_ydata(history_out[frame])
    ax_2d.set_title(f"Function Approximation (Step {current_step}/{iterations})")

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

    ax_3d.set_xlim([-3, 3])
    ax_3d.set_ylim([-3, 3])
    ax_3d.set_zlim([-3, 3]) 
    ax_3d.set_title("Support Evolution")

    ax_3d.view_init(elev=25, azim=-60)

    return line_net,

ani = animation.FuncAnimation(fig, update, frames=len(history_mu), interval=60, blit=False)

print("Saving the animation...")

# Save the animation in GIF format (DPI=150, FPS=15)
ani.save("sgd_evolution_quad.gif", writer='pillow', fps=15, dpi=150)
print("File sgd_evolution_quad.gif saved with success!")
