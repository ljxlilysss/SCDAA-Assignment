"""
Exercise 4.1: Policy Iteration with DGM
=========================================
Imports:
  - LQR class from Exercise 1.1 (for exact solution validation)
  - NetDGM from Exercise 2 (value network)
  - FFN from Exercise 2 (action network)
  - sample_interior, sample_terminal from Exercise 3
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Import from previous exercises 
from exercise_1_1_lqr_solver import LQR
from exercise_2_supervised_learning import NetDGM, FFN
from exercise_3_1_dgm import sample_interior, sample_terminal

EXP_NUM = 4
EXP_DIR = f"experiment{EXP_NUM}"
os.makedirs(EXP_DIR, exist_ok=True)

# ====================================================================
#  For reproduction
# ====================================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ====================================================================
# Wrapper: make NetDGM (expects concatenated tx) work with (t, x) calls
# ====================================================================
class ValueNetWrapper(nn.Module):
    """
    Wraps Exercise 2's NetDGM to accept separate (t, x) inputs.
    since NetDGM.forward expects a single concatenated tensor.
    """
    def __init__(self, in_dim=3, width=100, depth=3):
        super().__init__()
        self.net = NetDGM(in_dim=in_dim, width=width, depth=depth)

    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


# ====================================================================
# ActionNet with constant policy initialisation
# ====================================================================
class ActionNet(nn.Module):
    """
    Feed-forward network for control, with constant initialisation.
    Reuses Exercise 2's FFN structure but adds _init_constant_policy.
    """
    def __init__(self, in_dim=3, width=100, depth=2, out_dim=2):
        super().__init__()
        self.ffn = FFN(in_dim=in_dim, hidden_dim=width, out_dim=out_dim)

    def _init_constant_policy(self):
        """Initialise network to output constant (1, 1) everywhere."""
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 1.0)

    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        return self.ffn(inp)


# ====================================================================
# PIA Problem
# ====================================================================
class PIAProblem:
    def __init__(self, H, M, sigma, C, D, R, T, device='cpu'):
        self.H = torch.tensor(H, dtype=torch.float32, device=device)
        self.M = torch.tensor(M, dtype=torch.float32, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.C = torch.tensor(C, dtype=torch.float32, device=device)
        self.D = torch.tensor(D, dtype=torch.float32, device=device)
        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.T = float(T)
        self.device = device
        self.sigma_sigma_t = self.sigma @ self.sigma.T

    def terminal_value(self, x):
        return torch.sum((x @ self.R) * x, dim=1, keepdim=True)

    def pde_residual(self, value_net, action_net, t, x):
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)
        u = value_net(t, x)
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        grad_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        hessian = []
        for i in range(x.shape[1]):
            second = torch.autograd.grad(grad_x[:, i].sum(), x, create_graph=True)[0]
            hessian.append(second)
        hessian = torch.stack(hessian, dim=1)

        a = action_net(t, x)
        diff_term = 0.5 * torch.einsum('ij,bij->b', self.sigma_sigma_t, hessian).unsqueeze(1)
        drift_x = torch.einsum('bi,ij,bj->b', grad_x, self.H, x).unsqueeze(1)
        drift_a = torch.einsum('bi,ij,bj->b', grad_x, self.M, a).unsqueeze(1)
        running_x = torch.einsum('bi,ij,bj->b', x, self.C, x).unsqueeze(1)
        running_a = torch.einsum('bi,ij,bj->b', a, self.D, a).unsqueeze(1)
        return u_t + diff_term + drift_x + drift_a + running_x + running_a

    def hamiltonian(self, value_net, action_net, t, x):
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)
        u = value_net(t, x)
        grad_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        a = action_net(t, x)
        drift_x = torch.einsum('bi,ij,bj->b', grad_x, self.H, x).unsqueeze(1)
        drift_a = torch.einsum('bi,ij,bj->b', grad_x, self.M, a).unsqueeze(1)
        running_x = torch.einsum('bi,ij,bj->b', x, self.C, x).unsqueeze(1)
        running_a = torch.einsum('bi,ij,bj->b', a, self.D, a).unsqueeze(1)
        return drift_x + drift_a + running_x + running_a


# ====================================================================
# Evaluate against exact LQR
# ====================================================================
def evaluate_errors(value_net, action_net, exact, t_eval, x_eval, device='cpu'):
    value_net.eval()
    action_net.eval()
    with torch.no_grad():
        v_pred = value_net(t_eval, x_eval).cpu().numpy().squeeze()
        a_pred = action_net(t_eval, x_eval).cpu().numpy()
        # Adjust shapes for Exercise 1.1's LQR interface
        t_lqr = t_eval.squeeze(1)
        x_lqr = x_eval.unsqueeze(1)
        v_true = exact.value_function(t_lqr, x_lqr).cpu().numpy().squeeze()
        a_true = exact.optimal_control(t_lqr, x_lqr).cpu().numpy()
    v_err = np.mean(np.abs(v_pred - v_true) / np.maximum(1.0, np.abs(v_true)))
    a_err = np.mean(np.linalg.norm(a_pred - a_true, axis=1)
                    / np.maximum(1.0, np.linalg.norm(a_true, axis=1)))
    return float(v_err), float(a_err)


# ====================================================================
# Training functions
# ====================================================================
def train_policy_evaluation(problem, value_net, action_net, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        value_net.train()
        t_in, x_in = sample_interior(batch_size, problem.T, device=problem.device)
        t_T, x_T = sample_terminal(batch_size, problem.T, device=problem.device)
        residual = problem.pde_residual(value_net, action_net, t_in, x_in)
        boundary = value_net(t_T, x_T) - problem.terminal_value(x_T)
        loss = torch.mean(residual ** 2) + torch.mean(boundary ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


def train_policy_improvement(problem, value_net, action_net, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(action_net.parameters(), lr=lr)
    history = []
    for p in value_net.parameters():
        p.requires_grad_(False)
    for _ in range(epochs):
        action_net.train()
        t_in, x_in = sample_interior(batch_size, problem.T, device=problem.device)
        ham = problem.hamiltonian(value_net, action_net, t_in, x_in)
        loss = torch.mean(ham)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    for p in value_net.parameters():
        p.requires_grad_(True)
    return history


# ====================================================================
# Main PIA loop
# ====================================================================
def run_pia(problem, exact, n_policy_iter=10, eval_epochs=500,
            improve_epochs=600, batch_size=1024, lr_val=1e-3, lr_act=1e-3):

    value_net = ValueNetWrapper(in_dim=3, width=100, depth=3).to(problem.device)
    action_net = ActionNet(in_dim=3, width=100, depth=2).to(problem.device)
    action_net._init_constant_policy()  # Initialise to output (1, 1)

    # Fixed evaluation points
    t_eval = torch.rand(256, 1, device=problem.device) * problem.T
    x_eval = -3.0 + 6.0 * torch.rand(256, 2, device=problem.device)

    eval_loss_hist = []
    improve_loss_hist = []
    value_err_hist = []
    action_err_hist = []

    for k in range(n_policy_iter):
        print(f"\n--- Policy Iteration {k+1}/{n_policy_iter} ---")

        eval_hist = train_policy_evaluation(
            problem, value_net, action_net, eval_epochs, batch_size, lr_val)
        print(f"  Evaluation done, final loss = {eval_hist[-1]:.6e}")

        improve_hist = train_policy_improvement(
            problem, value_net, action_net, improve_epochs, batch_size, lr_act)
        print(f"  Improvement done, final Hamiltonian = {improve_hist[-1]:.6e}")

        v_err, a_err = evaluate_errors(
            value_net, action_net, exact, t_eval, x_eval, problem.device)

        eval_loss_hist.extend(eval_hist)
        improve_loss_hist.extend(improve_hist)
        value_err_hist.append(v_err)
        action_err_hist.append(a_err)
        print(f"  Value error = {v_err:.6e}, Action error = {a_err:.6e}")

    return (value_net, action_net, eval_loss_hist, improve_loss_hist,
            value_err_hist, action_err_hist, t_eval, x_eval)


# ====================================================================
# Plotting
# ====================================================================
def save_plots(eval_loss_hist, improve_loss_hist, value_err_hist, action_err_hist):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(eval_loss_hist)
    ax.set_xlabel('Evaluation step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Exercise 4.1: Policy Evaluation Loss', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'exercise_4_1_pia_eval_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: exercise_4_1_pia_eval_loss.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(improve_loss_hist)
    ax.set_xlabel('Improvement step', fontsize=12)
    ax.set_ylabel('Hamiltonian', fontsize=12)
    ax.set_title('Exercise 4.1: Policy Improvement Objective', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'exercise_4_1_pia_improve_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: exercise_4_1_pia_improve_loss.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.arange(1, len(value_err_hist) + 1)
    ax.semilogy(xs, value_err_hist, 'bo-', linewidth=2, markersize=7, label='Value error')
    ax.semilogy(xs, action_err_hist, 'rs-', linewidth=2, markersize=7, label='Action error')
    ax.set_xlabel('Policy iteration', fontsize=12)
    ax.set_ylabel('Mean relative error vs exact LQR', fontsize=12)
    ax.set_title('Exercise 4.1: Convergence to Exact LQR Solution', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'exercise_4_1_pia_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: exercise_4_1_pia_convergence.png")


def save_validation(value_net, action_net, exact, device='cpu'):
    t = torch.tensor([[0.0], [0.0], [0.25], [0.5], [0.75]],
                     dtype=torch.float32, device=device)
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5], [-1.0, 1.0], [1.5, 0.0]],
                     dtype=torch.float32, device=device)
    with torch.no_grad():
        v_pred = value_net(t, x).cpu().numpy().squeeze()
        a_pred = action_net(t, x).cpu().numpy()
        t_lqr = t.squeeze(1)
        x_lqr = x.unsqueeze(1)
        v_true = exact.value_function(t_lqr, x_lqr).cpu().numpy().squeeze()
        a_true = exact.optimal_control(t_lqr, x_lqr).cpu().numpy()
    with open(os.path.join(EXP_DIR,'exercise_4_1_validation.txt'), 'w', encoding='utf-8') as f:
        f.write('t,x1,x2,v_pred,v_true,v_rel_err,a1_pred,a2_pred,a1_true,a2_true,a_rel_err\n')
        for i in range(len(t)):
            v_rel = abs(v_pred[i] - v_true[i]) / max(1.0, abs(v_true[i]))
            a_rel = np.linalg.norm(a_pred[i] - a_true[i]) / max(1.0, np.linalg.norm(a_true[i]))
            f.write(
                f'{t[i,0].item():.2f},{x[i,0].item():.4f},{x[i,1].item():.4f},'
                f'{v_pred[i]:.6f},{v_true[i]:.6f},{v_rel:.6e},'
                f'{a_pred[i,0]:.6f},{a_pred[i,1]:.6f},'
                f'{a_true[i,0]:.6f},{a_true[i,1]:.6f},{a_rel:.6e}\n')
    print("Saved: exercise_4_1_validation.txt")


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Problem parameters
    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    D = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    T = 1.0

    # Exact LQR from Exercise 1.1 (for validation)
    exact = LQR(H, M, sigma, C, D, R, T)
    exact.solve_riccati(np.linspace(0.0, T, 10001))

    # PIA problem
    problem = PIAProblem(H, M, sigma, C, D, R, T, device=device)

    (value_net, action_net, eval_loss_hist, improve_loss_hist,
     value_err_hist, action_err_hist, _, _) = run_pia(
        problem, exact,
        n_policy_iter=10,
        eval_epochs=500,
        improve_epochs=600,
        batch_size=1024,
        lr_val=1e-3,
        lr_act=1e-3,
    )

    # Save outputs
    save_plots(eval_loss_hist, improve_loss_hist, value_err_hist, action_err_hist)
    save_validation(value_net, action_net, exact, device=device)
    torch.save(value_net.state_dict(), os.path.join(EXP_DIR,'exercise_4_1_value_net.pt'))
    torch.save(action_net.state_dict(), os.path.join(EXP_DIR,'exercise_4_1_action_net.pt'))

    print("\nExercise 4.1 complete!")
