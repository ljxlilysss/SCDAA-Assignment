"""
Exercise 1.1: LQR (Linear Quadratic Regulator) Solver

LQR problem:
  - The comtrolled SDE: dX_s = [H X_s + M alpha_s] ds + sigma dW_s,  s ∈ [t, T], X_t = x
  - Goal: Minimize J^alpha(t,x) = E[ ∫_t^T (X_s^T C X_s + alpha_s^T D alpha_s) ds + X_T^T R X_T ]

Analytical solution:
  - Value Function: v(t, x) = x^T S(t) x + ∫_t^T tr(sigma*sigma^T S(r)) dr
  - Optimal control:  a(t, x) = -D^{-1} M^T S(t) x
  - where S(t) satisfies the Riccati ODE:
      S'(r) = -2 H^T S(r) + S(r) M D^{-1} M^T S(r) - C,   S(T) = R

This module implements an LQR class, providing:
  1. solve_riccati()       -- solve Riccati ODE on a given time grid 
  2. value_function(t, x)  -- compute value function v(t, x)
  3. optimal_control(t, x) -- compute optimal control strategy a(t, x)
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

EXP_NUM = 1
EXP_DIR = f"experiment{EXP_NUM}"
os.makedirs(EXP_DIR, exist_ok=True)


class LQR:
    """LQR problem analytical solution solver"""
    def __init__(self, H, M, sigma, C, D, R, T):
        # Store problem matrices as numpy arrays
        self.H = np.array(H, dtype=np.float64)       
        self.M = np.array(M, dtype=np.float64)      
        self.sigma = np.array(sigma, dtype=np.float64)  
        self.C = np.array(C, dtype=np.float64)        
        self.D = np.array(D, dtype=np.float64)        
        self.R = np.array(R, dtype=np.float64)  
        self.T = float(T)

        # Precompute frequently used quantities
        self.D_inv = np.linalg.inv(self.D)
        self.D_inv_MT = self.D_inv @ self.M.T
        self.M_Dinv_MT = self.M @ self.D_inv_MT
        self.sigma_sigmaT = self.sigma @ self.sigma.T

        # Riccati solution storage (filled after solve_riccati)
        self._time_grid = None        
        self._S_values = None         
        self._sol_dense = None         
    
    # ================================================================
    # 1. solve Riccati ODE
    # ================================================================
    def solve_riccati(self, time_grid):
        """
        Solve Riccati ODE on given time grid 
        Since it is a terminal value problem, we convert it to initial value via time reversal
        """
        # Convert to numpy if input is torch tensor
        if isinstance(time_grid, torch.Tensor):
            time_grid_np = time_grid.detach().cpu().numpy()
        else:
            time_grid_np = np.array(time_grid, dtype=np.float64)

        self._time_grid = time_grid_np
        H_T = self.H.T
        M_Dinv_MT = self.M_Dinv_MT
        C_mat = self.C

        def riccati_rhs_reversed(tau, y):
            """RHS of Riccati ODE (time-reversed) as flat vector for scipy solver"""
            S = y.reshape(2, 2)
            dSdtau = 2.0 * H_T @ S - S @ M_Dinv_MT @ S + C_mat
            return dSdtau.ravel()

        # Initial condition (S(T) = R)
        y0 = self.R.ravel()
        tau_span = (0.0, self.T)
        tau_eval = self.T - time_grid_np
        tau_eval_sorted = np.sort(tau_eval)

        # Solve ODE for discrete time grid
        sol = solve_ivp(
            riccati_rhs_reversed,
            tau_span,
            y0,
            method='RK45',
            t_eval=tau_eval_sorted,
            rtol=1e-12,
            atol=1e-12,      
            max_step=0.001,  
        )

        if not sol.success:
            raise RuntimeError(f"Riccati ODE solver failed: {sol.message}")

        S_flat = sol.y[:, ::-1]
        N = len(time_grid_np)
        self._S_values = S_flat.T.reshape(N, 2, 2)

        # Solve again for dense output (interpolation)
        sol_dense = solve_ivp(
            riccati_rhs_reversed,
            (0.0, self.T),
            y0,
            method='RK45',
            dense_output=True,
            rtol=1e-12,
            atol=1e-12,
            max_step=0.001,
        )
        if not sol_dense.success:
            raise RuntimeError(f"Riccati ODE dense solver failed: {sol_dense.message}")

        self._sol_dense = sol_dense
        return self._S_values

    def _get_S_at_times(self, t_np):
        """Get S(t) at arbitrary time points via dense interpolation"""
        tau = self.T - t_np
        S_flat = self._sol_dense.sol(tau)
        if S_flat.ndim == 1:
            return S_flat.reshape(2, 2)
        else:
            return S_flat.T.reshape(-1, 2, 2)
        
    # ================================================================
    # 2.Value function v(t, x)
    # ================================================================
    def value_function(self, t, x):
        """Compute value function v(t, x) = x^T S(t) x + ∫_t^T tr(sigma*sigma^T S(r)) dr"""
        t_np = t.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        batch_size = t_np.shape[0]

        # Quadratic term: x^T S(t) x
        S_at_t = self._get_S_at_times(t_np)  
        x_2d = x_np[:, 0, :]  
        quadratic_term = np.einsum('ij,ijk,ik->i', x_2d, S_at_t, x_2d)
       
        # Integral term: ∫_t^T tr(σ σ^T S(r)) dr (trapezoidal rule)
        N_quad = 10000  
        r_grid = np.linspace(0, self.T, N_quad + 1)
        S_on_grid = self._get_S_at_times(r_grid)
        trace_vals = np.einsum('ij,kij->k', self.sigma_sigmaT, S_on_grid)

        dr = r_grid[1] - r_grid[0]
        cumulative_integral = np.cumsum((trace_vals[:-1] + trace_vals[1:]) / 2.0 * dr)
        cumulative_integral = np.concatenate([[0.0], cumulative_integral])  

        total_integral = cumulative_integral[-1] 
        integral_0_to_t = np.interp(t_np, r_grid, cumulative_integral) 
        integral_t_to_T = total_integral - integral_0_to_t  
        v_np = quadratic_term + integral_t_to_T 

        # Convert back to torch tensor
        v_torch = torch.tensor(v_np, dtype=x.dtype, device=x.device).unsqueeze(1)
        return v_torch
    

    # ================================================================
    # 3. Optimal a(t, x)
    # ================================================================
    def optimal_control(self, t, x):
        """Calculate optimal control a(t, x) = -D^{-1} M^T S(t) x"""
        # Convert to numpy
        t_np = t.detach().cpu().numpy()   
        x_np = x.detach().cpu().numpy()   

        # Get S(t) and reshape x
        S_at_t = self._get_S_at_times(t_np)  
        x_2d = x_np[:, 0, :]  

        # Compute optimal control
        a_np = -np.einsum('jk,ikl,il->ij', self.D_inv_MT, S_at_t, x_2d)

        # Convert back to torch tensor
        a_torch = torch.tensor(a_np, dtype=x.dtype, device=x.device)
        return a_torch


# ====================================================================
# Test
# ====================================================================
if __name__ == "__main__":
    # Test parameters
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.array([[1.0, 0.0], [0.0, 1.0]])
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    T = 1.0

    # Initialize LQR and solve Riccati ODE
    lqr = LQR(H, M, sigma, C, D, R_mat, T)
    N_time = 1000
    time_grid = np.linspace(0, T, N_time + 1)
    S_values = lqr.solve_riccati(time_grid)

    # Test value function
    batch_size = 5
    t_test = torch.rand(batch_size) * T
    x_test = torch.randn(batch_size, 1, 2)
    v = lqr.value_function(t_test, x_test)

    # Test optimal control
    a = lqr.optimal_control(t_test, x_test)

    # Verify t=T conditions
    t_T = torch.ones(3) * T
    x_check = torch.tensor([[[1.0, 2.0]], [[3.0, -1.0]], [[-2.0, 0.5]]])
    a_at_T = lqr.optimal_control(t_T, x_check)
    v_at_T = lqr.value_function(t_T, x_check)

    # Plot 1: S(t) matrix entries over time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_grid, S_values[:, 0, 0], 'b-',  linewidth=2, label='S(t)[0,0]')
    ax.plot(time_grid, S_values[:, 0, 1], 'r--', linewidth=2, label='S(t)[0,1]')
    ax.plot(time_grid, S_values[:, 1, 0], 'g-.', linewidth=2, label='S(t)[1,0]')
    ax.plot(time_grid, S_values[:, 1, 1], 'm:',  linewidth=2, label='S(t)[1,1]')
    ax.set_xlabel('Time t')
    ax.set_ylabel('S(t) entries')
    ax.set_title('Riccati ODE Solution: S(t) Matrix Entries over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'exercise_1_2_riccati_solution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Value function heatmap at t=0
    N_grid = 50
    x1_range = np.linspace(-3, 3, N_grid)
    x2_range = np.linspace(-3, 3, N_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    t_batch = torch.zeros(N_grid * N_grid)
    x_batch = torch.zeros(N_grid * N_grid, 1, 2)
    for i in range(N_grid):
        for j in range(N_grid):
            idx = i * N_grid + j
            x_batch[idx, 0, 0] = X1[i, j]
            x_batch[idx, 0, 1] = X2[i, j]

    v_batch = lqr.value_function(t_batch, x_batch).numpy().reshape(N_grid, N_grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(X1, X2, v_batch, levels=20, cmap='viridis')
    plt.colorbar(cp, ax=ax, label='v(0, x)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Value Function v(0, x) at t = 0')
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR,'exercise_1_1_value_function_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()