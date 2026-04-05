"""
Exercise 1.2: LQR Monte Carlo Verification

Goal:
  Verify the analytical solution in Exercise 1.1 using Monte Carlo simulation.
  Substitute the optimal control into the SDE, simulate the path, 
  calculate the average cost, and compare it with the analytical value function.

SDE (after substituting with optimal control):
  dX_s = [H X_s + M a(s, X_s)] ds + alpha dW_s
  where a(s, x) = -D^{-1} M^T S(s) x

Cost value:
  J = ∫_0^T (X_s^T C X_s + a_s^T D a_s) ds + X_T^T R X_T

Error metric:
  Relative error = |v̂_MC - v_exact| / |v_exact|
  Rationale: The magnitude of the value function depends on x0 and the parameters; 
  the relative error is scale-invariant.

Two formats:
  1. Explicit Euler: X_{n+1} = [I + τ(H - M D^{-1} M^T S(t_n))] X_n + sigma ΔW_n
  2. Implicit Euler: X_{n+1} = [I - τ(H - M D^{-1} M^T S(t_{n+1}))]^{-1} (X_n + sigma ΔW_n)

Expected convergence rate:
  Experiment 1 (variable N_steps): Euler is a first-order method → Error ∝ τ = T/N_steps → log-log slope ≈ 1
  Experiment 2 (variable N_MC): MC error ∝ 1/√N_MC → log-log slope ≈ -0.5
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from exercise_1_1_lqr_solver import LQR

EXP_NUM = 1
EXP_DIR = f"experiment{EXP_NUM}"
os.makedirs(EXP_DIR, exist_ok=True)

# ====================================================================
# Explicit Euler MC simulator
# ====================================================================
def mc_explicit_euler(lqr, x0, N_steps, N_mc, seed=42):
    """Explicit Euler Monte Carlo Simulation."""
    rng = np.random.RandomState(seed)

    T = lqr.T
    tau = T / N_steps
    sqrt_tau = np.sqrt(tau)

    # Precompute transition matrix A_n for each time step
    time_grid = np.linspace(0, T, N_steps + 1)
    S_all = lqr._get_S_at_times(time_grid)

    I2 = np.eye(2)
    M_Dinv_MT = lqr.M_Dinv_MT
    H = lqr.H

    A_list = np.zeros((N_steps, 2, 2))
    for n in range(N_steps):
        drift_matrix = H - M_Dinv_MT @ S_all[n]
        A_list[n] = I2 + tau * drift_matrix

    # Precompute comprehensive cost matrix Q_n
    D_inv_MT = lqr.D_inv_MT
    Q_list = np.zeros((N_steps, 2, 2))
    for n in range(N_steps):
        G_n = D_inv_MT @ S_all[n]
        Q_list[n] = lqr.C + G_n.T @ lqr.D @ G_n

    sigma = lqr.sigma

    # Simulate in batches to save memory
    CHUNK = min(N_mc, 20000)
    total_cost_sum = 0.0
    n_processed = 0

    while n_processed < N_mc:
        batch = min(CHUNK, N_mc - n_processed)
        X = np.tile(x0, (batch, 1))
        running_cost = np.zeros(batch)

        for n in range(N_steps):
            # Calculate running cost
            running_cost += tau * np.einsum('ij,jk,ik->i', X, Q_list[n], X)
            # Generate Brownian increment
            dW = rng.randn(batch, 2) * sqrt_tau
            # Explicit Euler step
            X = X @ A_list[n].T + dW @ sigma.T

        # Calculate terminal cost
        terminal_cost = np.einsum('ij,jk,ik->i', X, lqr.R, X)
        total_cost_sum += np.sum(running_cost + terminal_cost)
        n_processed += batch

    v_mc = total_cost_sum / N_mc
    return v_mc

# ====================================================================
# Implicit Euler MC simulator
# ====================================================================
def mc_implicit_euler(lqr, x0, N_steps, N_mc, seed=42):
    """Implicit Euler Monte Carlo Simulator"""
    rng = np.random.RandomState(seed)

    T = lqr.T
    tau = T / N_steps
    sqrt_tau = np.sqrt(tau)

    # Precompute B_{n+1}^{-1} and Q_n
    time_grid = np.linspace(0, T, N_steps + 1)
    S_all = lqr._get_S_at_times(time_grid)

    I2 = np.eye(2)
    M_Dinv_MT = lqr.M_Dinv_MT
    H = lqr.H
    D_inv_MT = lqr.D_inv_MT

    B_inv_list = np.zeros((N_steps, 2, 2))
    for n in range(N_steps):
        drift_matrix = H - M_Dinv_MT @ S_all[n + 1]
        B = I2 - tau * drift_matrix
        B_inv_list[n] = np.linalg.inv(B)

    # Precompute Q_n
    Q_list = np.zeros((N_steps, 2, 2))
    for n in range(N_steps):
        G_n = D_inv_MT @ S_all[n]
        Q_list[n] = lqr.C + G_n.T @ lqr.D @ G_n

    sigma = lqr.sigma

    # Simulate in batches
    CHUNK = min(N_mc, 20000)
    total_cost_sum = 0.0
    n_processed = 0

    while n_processed < N_mc:
        batch = min(CHUNK, N_mc - n_processed)
        X = np.tile(x0, (batch, 1))
        running_cost = np.zeros(batch)

        for n in range(N_steps):
            # Calculate running cost
            running_cost += tau * np.einsum('ij,jk,ik->i', X, Q_list[n], X)
            # Generate Brownian increment
            dW = rng.randn(batch, 2) * sqrt_tau
            # Implicit Euler step
            rhs = X + dW @ sigma.T
            X = rhs @ B_inv_list[n].T

        # Calculate terminal cost
        terminal_cost = np.einsum('ij,jk,ik->i', X, lqr.R, X)
        total_cost_sum += np.sum(running_cost + terminal_cost)
        n_processed += batch

    v_mc = total_cost_sum / N_mc
    return v_mc


# ====================================================================
# Test 
# ====================================================================
if __name__ == "__main__":
    # Set parameters
    H = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
    M = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
    sigma = np.array([[0.5, 0.0],
                       [0.0, 0.5]])
    C = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
    D = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
    R_mat = np.array([[1.0, 0.0],
                       [0.0, 1.0]])
    T = 1.0

    lqr = LQR(H, M, sigma, C, D, R_mat, T)
    fine_grid = np.linspace(0, T, 10001)
    lqr.solve_riccati(fine_grid)

    # Analytical value function (reference value)
    x0 = np.array([1.0, 0.5])
    t_tensor = torch.tensor([0.0])
    x_tensor = torch.tensor(x0).unsqueeze(0).unsqueeze(0).float()
    v_exact = lqr.value_function(t_tensor, x_tensor).item()
    print(f"Analytical value function v(0, x0) = {v_exact:.10f}")
    print(f"initial state x0 = {x0}\n")

    # Experiment 1: Fix N_MC, vary N_steps (time discretization convergence)
    print("=" * 70)
    print("Experiment 1: Time Discretization Convergence (fixed N_MC=100,000)")
    print("=" * 70)

    N_mc_fixed = 100000
    N_steps_list = [1, 10, 50, 100, 500, 1000, 5000]

    errors_explicit_exp1 = []
    errors_implicit_exp1 = []

    for N_steps in N_steps_list:
        v_exp = mc_explicit_euler(lqr, x0, N_steps, N_mc_fixed)
        v_imp = mc_implicit_euler(lqr, x0, N_steps, N_mc_fixed)
        err_exp = abs(v_exp - v_exact) / abs(v_exact)
        err_imp = abs(v_imp - v_exact) / abs(v_exact)
        errors_explicit_exp1.append(err_exp)
        errors_implicit_exp1.append(err_imp)
        print(f"  N_steps={N_steps:5d} | "
              f"Explicit: v={v_exp:.6f}, err={err_exp:.2e} | "
              f"Implicit: v={v_imp:.6f}, err={err_imp:.2e}")

    # Experiment 2: Fix N_steps, vary N_MC (MC sample convergence)
    print("=" * 70)
    print("Experiment 2: Convergence of MC sample size (fixed N_steps=5,000)")
    print("=" * 70)

    N_steps_fixed = 5000
    N_mc_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

    errors_explicit_exp2 = []
    errors_implicit_exp2 = []

    for N_mc in N_mc_list:
        v_exp = mc_explicit_euler(lqr, x0, N_steps_fixed, N_mc)
        v_imp = mc_implicit_euler(lqr, x0, N_steps_fixed, N_mc)
        err_exp = abs(v_exp - v_exact) / abs(v_exact)
        err_imp = abs(v_imp - v_exact) / abs(v_exact)
        errors_explicit_exp2.append(err_exp)
        errors_implicit_exp2.append(err_imp)
        print(f"  N_MC={N_mc:6d} | "
              f"Explicit: v={v_exp:.6f}, err={err_exp:.2e} | "
              f"Implicit: v={v_imp:.6f}, err={err_imp:.2e}")

    # Log-Log plot 1: Time discretization convergence
    fig, ax = plt.subplots(figsize=(14, 6))
    dt_list = [T / N for N in N_steps_list]

    ax.loglog(dt_list, errors_explicit_exp1, 'bo-',
              label='Explicit Euler', linewidth=2, markersize=7)
    ax.loglog(dt_list, errors_implicit_exp1, 'rs-',
              label='Implicit Euler', linewidth=2, markersize=7)

    # Reference slope line: O(τ^1)
    dt_arr = np.array(dt_list)
    ref_idx = len(dt_list) // 2
    ref_line = errors_explicit_exp1[ref_idx] * (dt_arr / dt_arr[ref_idx]) ** 1.0
    ax.loglog(dt_arr, ref_line, 'k--', alpha=0.5, label='Slope 1 (reference)')

    ax.set_xlabel('Time step size τ = T/N', fontsize=12)
    ax.set_ylabel('Relative error', fontsize=12)
    ax.set_title('Exp 1: Time Discretisation Convergence\n(N_MC = 100,000)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'exercise_1_2_time_discretisation_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nExercise1.2: exercise_1_2_time_discretisation_convergence.png has saved!")

    # Log-Log plot 2: MC sample number convergence
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.loglog(N_mc_list, errors_explicit_exp2, 'bo-',
              label='Explicit Euler', linewidth=2, markersize=7)
    ax.loglog(N_mc_list, errors_implicit_exp2, 'rs-',
              label='Implicit Euler', linewidth=2, markersize=7)

    # Reference slope line: O(N^{-0.5})
    N_arr = np.array(N_mc_list, dtype=float)
    ref_idx2 = len(N_mc_list) // 2
    ref_line2 = errors_explicit_exp2[ref_idx2] * (N_arr / N_arr[ref_idx2]) ** (-0.5)
    ax.loglog(N_arr, ref_line2, 'k--', alpha=0.5, label='Slope -1/2 (reference)')

    ax.set_xlabel('Number of MC samples N_MC', fontsize=12)
    ax.set_ylabel('Relative error', fontsize=12)
    ax.set_title('Exp 2: MC Sample Convergence\n(N_steps = 5,000)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'exercise_1_2_mc_sample_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Exercise1.2: exercise_1_2_mc_sample_convergence.png has saved!")