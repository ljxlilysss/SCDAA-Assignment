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
from lqr_solver import LQR


# ====================================================================
# Explicit Euler MC simulator
# ====================================================================
def mc_explicit_euler(lqr, x0, N_steps, N_mc, seed=42):
    """
    Explicit Euler Monte Carlo Simulation.

    Derivation:
      Substituting a(t_n, X_n) = -D^{-1} M^T S(t_n) X_n into the explicit Euler formula:
      X_{n+1} = X_n + τ [H X_n + M(-D^{-1} M^T S(t_n) X_n)] + sigma ΔW_n
      = X_n + τ [H - M D^{-1} M^T S(t_n)] X_n + sigma ΔW_n= A_n X_n + sigma ΔW_n
      Where A_n = I + τ (H - M D^{-1} M^T S(t_n))

    To save memory, we process MC samples in chunks.
    
    Parameters:
      lqr: LQR object (already solved_riccati)
      x0: np.ndarray (2,), initial state
      N_steps: int, number of time steps
      N_mc: int, number of MC samples
      seed: int, random seed
    
    Returns:
      v_mc: float, MC estimate of v(0, x0)
    """
    rng = np.random.RandomState(seed)

    T = lqr.T
    tau = T / N_steps
    sqrt_tau = np.sqrt(tau)

    # Precompute the transition matrix for each time step. A_n
    # A_n = I + τ (H - M D^{-1} M^⊤ S(t_n)), shape (2,2)
    time_grid = np.linspace(0, T, N_steps + 1)
    S_all = lqr._get_S_at_times(time_grid)  # shape (N_steps+1, 2, 2)

    I2 = np.eye(2)
    M_Dinv_MT = lqr.M_Dinv_MT
    H = lqr.H

    A_list = np.zeros((N_steps, 2, 2))
    for n in range(N_steps):
        drift_matrix = H - M_Dinv_MT @ S_all[n]
        A_list[n] = I2 + tau * drift_matrix

    # Pre-calculate control gain matrix G_n = D^{-1} M^⊤ S(t_n)
    # a_n = -G_n X_n, 且 a^⊤ D a = X^⊤ G_n^⊤ D G_n X = X^⊤ (S M D^{-1} M^⊤ S) X
    # Precompute Q_n = C + S(t_n) M D^{-1} M^⊤ S(t_n) (Comprehensive Cost Matrix)
    # running_cost_n = X^⊤ Q_n X
    # a^⊤ D a = (G X)^⊤ D (G X) = X^⊤ G^⊤ D G X
    # G = D^{-1} M^⊤ S, G^⊤ D G = S M D^{-⊤} D D^{-1} M^⊤ S = S M D^{-1} M^⊤ S
    D_inv_MT = lqr.D_inv_MT
    Q_list = np.zeros((N_steps, 2, 2))
    for n in range(N_steps):
        G_n = D_inv_MT @ S_all[n]  # shape (2,2)
        Q_list[n] = lqr.C + G_n.T @ lqr.D @ G_n

    sigma = lqr.sigma

    # Simulate in batches to save memory.
    CHUNK = min(N_mc, 20000)  # Maximum of 20,000 paths per batch
    total_cost_sum = 0.0

    n_processed = 0
    while n_processed < N_mc:
        batch = min(CHUNK, N_mc - n_processed)

        # X shape: (batch, 2)
        X = np.tile(x0, (batch, 1))
        running_cost = np.zeros(batch)

        for n in range(N_steps):
            # Running cost: X^⊤ Q_n X (include state costs and control costs)
            # einsum: for each path i, cost_i = Σ_jk X[i,j] Q[j,k] X[i,k]
            running_cost += tau * np.einsum('ij,jk,ik->i', X, Q_list[n], X)

            # Generate Brownian increment ΔW ~ N(0, τ I)
            dW = rng.randn(batch, 2) * sqrt_tau

            # Explicit Euler step：X_{n+1} = A_n X_n + σ ΔW_n
            X = X @ A_list[n].T + dW @ sigma.T

        # Terminal: X_T^⊤ R X_T
        terminal_cost = np.einsum('ij,jk,ik->i', X, lqr.R_mat, X)
        total_cost_sum += np.sum(running_cost + terminal_cost)
        n_processed += batch

    v_mc = total_cost_sum / N_mc
    return v_mc


# Implicit Euler MC simulator
def mc_implicit_euler(lqr, x0, N_steps, N_mc, seed=42):
    """
    Implicit Euler Monte Carlo Simulator

    Derivation:
      X_{n+1} = X_n + τ [H X_{n+1} + M a(t_{n+1}, X_{n+1})] + sigma ΔW_n
      substitute with a(t, x) = -D^{-1} M^T S(t) x:
      X_{n+1} = X_n + τ [H - M D^{-1} M^T S(t_{n+1})] X_{n+1} + sigma ΔW_n

    organize:
      [I - τ (H - M D^{-1} M^T S(t_{n+1}))] X_{n+1} = X_n + sigma ΔW_n
      let B_{n+1} = I - τ (H - M D^{-1} M^T S(t_{n+1}))
      and get X_{n+1} = B_{n+1}^{-1} (X_n + sigma ΔW_n)

    Each step requires solving a 2*2 system of linear equations, but since B does not change with the path,
    B^{-1} can be pre-calculated, and then only matrix multiplication is needed.

    Paremeters/Return: the same as mc_explicit_euler
    """
    rng = np.random.RandomState(seed)

    T = lqr.T
    tau = T / N_steps
    sqrt_tau = np.sqrt(tau)

    # precalculate B_{n+1}^{-1} and comprehensive cost matrix Q_n
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

    # running cost Q_n = C + G_n^⊤ D G_n
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
            # Operating cost (calculated at X_n, using the left-endpoint rule)
            running_cost += tau * np.einsum('ij,jk,ik->i', X, Q_list[n], X)

            # Brownian increment
            dW = rng.randn(batch, 2) * sqrt_tau

            # Implicit Euler step: X_{n+1} = B_{n+1}^{-1} (X_n + σ ΔW_n)
            rhs = X + dW @ sigma.T
            X = rhs @ B_inv_list[n].T

        terminal_cost = np.einsum('ij,jk,ik->i', X, lqr.R_mat, X)
        total_cost_sum += np.sum(running_cost + terminal_cost)
        n_processed += batch

    v_mc = total_cost_sum / N_mc
    return v_mc


# ====================================================================
# Test/Example usage
# ====================================================================
if __name__ == "__main__":

    # ==============================================================
    # 1. Set parameters
    # ==============================================================
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

    # ==============================================================
    # 2. Analytical value function (reference value)
    # ==============================================================
    x0 = np.array([1.0, 0.5])
    t_tensor = torch.tensor([0.0])
    x_tensor = torch.tensor(x0).unsqueeze(0).unsqueeze(0).float()
    v_exact = lqr.value_function(t_tensor, x_tensor).item()
    print(f"Analytical value function v(0, x0) = {v_exact:.10f}")
    print(f"initial state x0 = {x0}\n")

    # ==============================================================
    # Experiment 1: Fix N_MC = 10^5, vary N_steps
    # Expected value: Error ∝ τ = T/N_steps → log-log Slope ≈ 1
    # ==============================================================
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

    # ==============================================================
    # Experiment 2: Fix N_steps=5000, vary N_MC
    # Expected: Error ∝ 1/√N_MC → log-log Slope ≈ -0.5
    # ==============================================================
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

    # ==============================================================
    # Log-Log plot
    # ==============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Figure 1: Time discretization convergence
    ax = axes[0]
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

    # Figure 2: MC sample number convergence 
    ax = axes[1]

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
    plt.savefig('exercise_1_2_convergence.png',
                dpi=150, bbox_inches='tight')
   
