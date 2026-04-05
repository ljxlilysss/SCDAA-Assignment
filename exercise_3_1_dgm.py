"""
Exercise 3.1: Deep Galerkin Method for Linear PDE
===================================================
Imports:
  - LQR class from Exercise 1.1 (for MC validation)
  - NetDGM network from Exercise 2 (for DGM approximation)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from exercise_1_1_lqr_solver import LQR
from exercise_2_supervised_learning import NetDGM

EXP_NUM = 3
EXP_DIR = f"experiment{EXP_NUM}"
os.makedirs(EXP_DIR, exist_ok=True)

# ====================================================================
#  For reproduction
# ====================================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

# ====================================================================
# Linear PDE Problem (constant control alpha, torch tensors for autograd)
# ====================================================================
class LinearPDEProblem:
    """
    Encapsulates the linear PDE with constant control alpha = (1, 1).
    Uses torch tensors internally because PDE residual computation
    requires automatic differentiation.
    """
    def __init__(self, H, M, sigma, C, D, R, T, alpha):
        self.H = torch.tensor(H, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)
        self.T = float(T)
        self.alpha = torch.tensor(alpha, dtype=torch.float32).view(2, 1)
        self.sigma_sigma_t = self.sigma @ self.sigma.T
        self.alpha_D_alpha = (self.alpha.T @ self.D @ self.alpha).item()

    def terminal_value(self, x):
        return torch.sum((x @ self.R) * x, dim=1, keepdim=True)

    def residual(self, net, t, x):
        """
        Compute PDE residual using autograd.
        NetDGM from Exercise 2 expects concatenated input (batch, 3).
        """
        t.requires_grad_(True)
        x.requires_grad_(True)

        # NetDGM expects concatenated input (batch, 3)
        inp = torch.cat([t, x], dim=1)
        u = net(inp)

        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        grad_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        hessian = []
        for i in range(x.shape[1]):
            second = torch.autograd.grad(grad_x[:, i].sum(), x, create_graph=True)[0]
            hessian.append(second)
        hessian = torch.stack(hessian, dim=1)

        diff_term = 0.5 * torch.einsum('ij,bij->b', self.sigma_sigma_t, hessian).unsqueeze(1)
        drift_x = torch.sum((x @ self.H.T) * grad_x, dim=1, keepdim=True)
        drift_a = torch.sum((self.alpha.view(1, 2) @ self.M.T) * grad_x, dim=1, keepdim=True)
        running_x = torch.sum((x @ self.C) * x, dim=1, keepdim=True)
        running_a = torch.full_like(running_x, self.alpha_D_alpha)

        return u_t + diff_term + drift_x + drift_a + running_x + running_a


# ====================================================================
# Sampling functions
# ====================================================================
def sample_interior(batch_size, T, x_low=-3.0, x_high=3.0, device='cpu'):
    t = T * torch.rand(batch_size, 1, device=device)
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


def sample_terminal(batch_size, T, x_low=-3.0, x_high=3.0, device='cpu'):
    t = torch.full((batch_size, 1), T, device=device)
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


# ====================================================================
# Monte Carlo Benchmark for Constant Control
# ====================================================================
def mc_constant_control(lqr, x0, alpha, N_steps, N_mc, t0=0.0, seed=42):
    """
    MC simulation with constant control alpha.
    Same structure as Exercise 1.2's mc_explicit_euler, but control is fixed.
    """
    rng = np.random.RandomState(seed)

    T = lqr.T
    tau = (T - t0) / N_steps
    sqrt_tau = np.sqrt(tau)

    if tau <= 0:
        return float(x0 @ lqr.R_mat @ x0)

    H = lqr.H
    M = lqr.M
    sigma = lqr.sigma
    C = lqr.C

    alpha = np.array(alpha, dtype=np.float64)
    alpha_D_alpha = alpha @ lqr.D @ alpha
    M_alpha = M @ alpha

    CHUNK = min(N_mc, 20000)
    total_cost_sum = 0.0
    n_processed = 0

    while n_processed < N_mc:
        batch = min(CHUNK, N_mc - n_processed)
        X = np.tile(x0, (batch, 1))
        running_cost = np.zeros(batch)

        for n in range(N_steps):
            running_cost += tau * (np.einsum('ij,jk,ik->i', X, C, X) + alpha_D_alpha)
            dW = rng.randn(batch, 2) * sqrt_tau
            X = X + tau * (X @ H.T + M_alpha) + dW @ sigma.T

        terminal_cost = np.einsum('ij,jk,ik->i', X, lqr.R, X)
        total_cost_sum += np.sum(running_cost + terminal_cost)
        n_processed += batch

    return total_cost_sum / N_mc


# ====================================================================
# Validation
# ====================================================================
def build_validation_set(lqr, alpha):
    """Build validation points with MC reference values using Exercise 1.1's LQR."""
    pts = [
        (0.0,  np.array([0.0, 0.0])),
        (0.0,  np.array([1.0, 0.5])),
        (0.0,  np.array([-1.0, 1.0])),
        (0.25, np.array([0.5, -0.5])),
        (0.25, np.array([1.5, 0.0])),
        (0.5,  np.array([0.0, 1.0])),
        (0.5,  np.array([-1.0, -1.0])),
        (0.75, np.array([0.5, 0.5])),
        (0.75, np.array([-0.5, 1.5])),
    ]
    refs = []
    for i, (t0, x0) in enumerate(pts):
        n_steps = max(200, int((lqr.T - t0) * 800))
        ref = mc_constant_control(lqr, x0, alpha,
                                  N_steps=n_steps, N_mc=5000, t0=t0, seed=2026 + i)
        refs.append(ref)
    return pts, np.array(refs)


def evaluate_mc_error(net, pts, refs, device='cpu'):
    """Compute mean relative error of network predictions vs MC reference."""
    net.eval()
    t = torch.tensor([[p[0]] for p in pts], dtype=torch.float32, device=device)
    x = torch.tensor(np.stack([p[1] for p in pts]), dtype=torch.float32, device=device)
    with torch.no_grad():
        inp = torch.cat([t, x], dim=1)
        pred = net(inp).squeeze(1).cpu().numpy()
    denom = np.maximum(1.0, np.abs(refs))
    return float(np.mean(np.abs(pred - refs) / denom))


# ====================================================================
# Training
# ====================================================================
def train_dgm(problem, pts, refs, epochs=1000, batch_size=1024, lr=1e-3, device='cpu'):
    """
    Train DGM network to solve the linear PDE.
    pts and refs are pre-computed MC validation data from build_validation_set.
    """
    net = NetDGM(in_dim=3, width=100, depth=3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_history = []
    error_steps = []
    error_history = []

    for epoch in range(1, epochs + 1):
        net.train()
        t_in, x_in = sample_interior(batch_size, problem.T, device=device)
        t_bd, x_bd = sample_terminal(batch_size, problem.T, device=device)

        # PDE residual (residual handles concatenation internally)
        residual = problem.residual(net, t_in, x_in)

        # Terminal condition: NetDGM expects concatenated input
        inp_bd = torch.cat([t_bd, x_bd], dim=1)
        boundary = net(inp_bd) - problem.terminal_value(x_bd)

        loss = torch.mean(residual ** 2) + torch.mean(boundary ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 100 == 0 or epoch == 1:
            err = evaluate_mc_error(net, pts, refs, device=device)
            error_steps.append(epoch)
            error_history.append(err)
            print(f"epoch={epoch:4d}  loss={loss.item():.6e}  mc_error={err:.6e}")

    return net, loss_history, error_steps, error_history


# ====================================================================
# Plotting
# ====================================================================
def plot_training(loss_history, error_steps, error_history):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(loss_history)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training loss', fontsize=12)
    ax.set_title('Exercise 3.1: DGM Training Loss', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'Exercise_3_1_dgm_training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: Exercise_3_1_dgm_training_loss.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(error_steps, error_history, 'bo-', linewidth=2, markersize=7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean relative error vs MC', fontsize=12)
    ax.set_title('Exercise 3.1: DGM Error Against Monte Carlo Reference', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'Exercise_3_1_dgm_mc_error.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: Exercise_3_1_dgm_mc_error.png")


def save_predictions(net, pts, refs, device='cpu'):
    t = torch.tensor([[p[0]] for p in pts], dtype=torch.float32, device=device)
    x = torch.tensor(np.stack([p[1] for p in pts]), dtype=torch.float32, device=device)
    with torch.no_grad():
        inp = torch.cat([t, x], dim=1)
        pred = net(inp).squeeze(1).cpu().numpy()

    with open(os.path.join(EXP_DIR,'exercise_3_1_validation.txt'), 'w', encoding='utf-8') as f:
        f.write('t,x1,x2,dgm_value,mc_value,rel_error\n')
        for (t0, x0), p, r in zip(pts, pred, refs):
            rel = abs(p - r) / max(1.0, abs(r))
            f.write(f'{t0:.2f},{x0[0]:.4f},{x0[1]:.4f},{p:.6f},{r:.6f},{rel:.6e}\n')
    print("Saved: exercise_3_1_validation.txt")


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Problem parameters (same as all exercises)
    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    D = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    alpha = np.array([1.0, 1.0], dtype=np.float32)
    T = 1.0

    # LinearPDEProblem for DGM training (torch tensors, computes PDE residual)
    problem = LinearPDEProblem(H, M, sigma, C, D, R, T, alpha)

    # LQR object for MC validation (numpy, from Exercise 1.1)
    lqr = LQR(H, M, sigma, C, D, R, T)
    lqr.solve_riccati(np.linspace(0, T, 10001))

    # Build MC reference values
    alpha_np = np.array([1.0, 1.0])
    pts, refs = build_validation_set(lqr, alpha_np)

    # Train DGM network
    net, loss_history, error_steps, error_history = train_dgm(
        problem, pts, refs,
        epochs=1000, batch_size=1024, lr=1e-3, device=device,
    )

    # Save outputs
    plot_training(loss_history, error_steps, error_history)
    save_predictions(net, pts, refs, device=device)
    torch.save(net.state_dict(), os.path.join(EXP_DIR,'exercise_3_1_dgm_weights.pt'))

    print("\nExercise 3.1 complete!")
