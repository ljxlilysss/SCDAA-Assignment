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


class LQR:
    """
    LQR problem analaytical solution solver

    Parameters: (2*2 Matrix):
        H : np.ndarray (2,2) -- drift matrix in state dynamics
        M : np.ndarray (2,2) -- control input matrix
        sigma : np.ndarray (2,2) -- diffusion / volatility matrix
        C : np.ndarray (2,2) -- running cost, state weight, C ≥ 0
        D : np.ndarray (2,2) -- running cost, control weight, D > 0
        R : np.ndarray (2,2) -- terminal cost weight, R ≥ 0
        T : float            -- terminal time
    """

    def __init__(self, H, M, sigma, C, D, R_mat, T):
        
        # Store all problem matrices as numpy arrays (used in Riccati solver)
        self.H = np.array(H, dtype=np.float64)       
        self.M = np.array(M, dtype=np.float64)      
        self.sigma = np.array(sigma, dtype=np.float64)  
        self.C = np.array(C, dtype=np.float64)        
        self.D = np.array(D, dtype=np.float64)        
        self.R_mat = np.array(R_mat, dtype=np.float64)  
        self.T = float(T)

        # Precompute frequently used quantities

        # D^{-1}: Inverse of the control cost matrix
        self.D_inv = np.linalg.inv(self.D)

        # D^{-1} M^⊤: will be used in optimal control a = -D^{-1} M^⊤ S(t) x and Riccati ODE
        self.D_inv_MT = self.D_inv @ self.M.T  # shape (2,2)

        # M D^{-1} M^⊤: will be used in Riccati ODE  "gain" part
        self.M_Dinv_MT = self.M @ self.D_inv_MT  # shape (2,2)

        # σ σ^⊤: will be used in tr(σσ^⊤ S(r)) 
        self.sigma_sigmaT = self.sigma @ self.sigma.T  # shape (2,2)

        # To store Riccati solution (Will be filled after solve_riccati)
        self._time_grid = None        # 1-D array of time points (ascending)
        self._S_values = None         # list/array of 2x2 S matrices at each time point, shape (N, 2, 2)
        self._S_interp = None         # scipy solution object used for interpolation

    # ================================================================
    # 1. solve Riccati ODE
    # ================================================================
    def solve_riccati(self, time_grid):
        """
        Solve Riccati ODE on given time grid:
            S'(r) = -2 H^T S(r) + S(r) M D^{-1} M^T S(r) - C
            S(T) = R

        Since it is a terminal value problem), We integrate backwards from T to 0.

        Parameter:
            time_grid: numpy array or torch tensor, like [t_0, t_1, ..., t_N]
                       and t_0 <= t_1 <= ... <= t_N = T 

        Strategy:
            We use scipy.integrate.solve_ivp RK45 method With very strict error 
            tolerances (rtol=1e-12, atol=1e-12) to ensure the accuracy of
            the numerical solution of the Riccati ODE.

        Some details:
            - Since S(r) is a 2*2 symmetric matrix
            - We flatten it into a vector of length 4 and pass it to the ODE solver.
            - Since it's a terminal value problem S(T)=R, we perform time reversal: let τ = T - r,
              and get dS/dτ = 2H^T S - S M D^{-1} M^T S + C, S(τ=0) = R
        """
        # if input is torch tensor，transform it into numpy
        if isinstance(time_grid, torch.Tensor):
            time_grid_np = time_grid.detach().cpu().numpy()
        else:
            time_grid_np = np.array(time_grid, dtype=np.float64)

        self._time_grid = time_grid_np
        H_T = self.H.T  # H's transpose
        M_Dinv_MT = self.M_Dinv_MT
        C_mat = self.C

        def riccati_rhs_reversed(tau, y):
            """
            return:
            Right-hand side of the Riccati ODE written as a flat vector equation
            so that scipy's ODE solvers can handle it.
            Parameters:
            tau: τ = T - r
            y:   (4,) array — current S matrix(2*2) flattened row-major
            """
            S = y.reshape(2, 2)

            # dS/dτ = 2 H^⊤ S - S (M D^{-1} M^⊤) S + C
            dSdtau = 2.0 * H_T @ S - S @ M_Dinv_MT @ S + C_mat

            return dSdtau.ravel()

        # Initial condition: S(T) = R, when τ=0, S = R
        y0 = self.R_mat.ravel()
        # integral span from 0 to T
        tau_span = (0.0, self.T)
        # reverse
        tau_eval = self.T - time_grid_np
        tau_eval_sorted = np.sort(tau_eval)  # ascending

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

        S_flat = sol.y[:, ::-1]  # shape (4, N)

        # reshape it to matrix
        N = len(time_grid_np)
        self._S_values = S_flat.T.reshape(N, 2, 2)  # shape (N, 2, 2)

        # Solve again, this time using dense_output for interpolation at any time points.
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

        self._sol_dense = sol_dense  # store dense output for interpolations

        return self._S_values

    def _get_S_at_times(self, t_np):
        """
        Obtain the value of S(t) at any time point t (numpy array).
        Use dense_output for high-precision interpolation.

        Paremeters:
            t_np: numpy array, shape (batch,), time points

        Return:
            S_vals: numpy array, shape (batch, 2, 2)
        """

        # τ = T - t
        tau = self.T - t_np
        S_flat = self._sol_dense.sol(tau)  # shape (4, len(tau)) or (4,) if scalar

        if S_flat.ndim == 1:
            return S_flat.reshape(2, 2)
        else:
            return S_flat.T.reshape(-1, 2, 2)  # shape (batch, 2, 2)

    # 2. Value Function v(t, x)
    def value_function(self, t, x):
        """
        Compute value function:
            v(t, x) = x^T S(t) x + ∫_t^T tr(sigma*sigma^T S(r)) dr
        Parameters:
            t: torch.Tensor, shape (batch_size,) time points
            x: torch.Tensor, shape (batch_size, 1, 2) State vectors (2-dimensional, stored as row vectors)
        Return:
            v: torch.Tensor, shape (batch_size, 1)
               Value function at each (t, x) pair.
        """

        t_np = t.detach().cpu().numpy()                    # shape (batch,)
        x_np = x.detach().cpu().numpy()                    # shape (batch, 1, 2)

        batch_size = t_np.shape[0]

        # The first item: x^⊤ S(t) x
        S_at_t = self._get_S_at_times(t_np)  # shape (batch, 2, 2)

        # x_np shape: (batch, 1, 2) 
        x_2d = x_np[:, 0, :]  # shape (batch, 2)

        # quadratic_term[i] = sum_j sum_k x_2d[i,j] * S_at_t[i,j,k] * x_2d[i,k]
        quadratic_term = np.einsum('ij,ijk,ik->i', x_2d, S_at_t, x_2d)  # shape (batch,)
       
        N_quad = 10000  
        r_grid = np.linspace(0, self.T, N_quad + 1)  # shape (N_quad+1,)

        S_on_grid = self._get_S_at_times(r_grid)  # shape (N_quad+1, 2, 2)
        ssT = self.sigma_sigmaT  # (2,2)
        trace_vals = np.einsum('ij,kij->k', ssT, S_on_grid)  # shape (N_quad+1,)

        # cumulative_integral[k] = ∫_0^{r_k} tr(σσ^⊤ S(r)) dr
        dr = r_grid[1] - r_grid[0]
        cumulative_integral = np.cumsum(
            (trace_vals[:-1] + trace_vals[1:]) / 2.0 * dr
        )
        cumulative_integral = np.concatenate([[0.0], cumulative_integral])  # shape (N_quad+1,)

        # ∫_t^T tr(σσ^⊤ S(r)) dr = ∫_0^T ... dr - ∫_0^t ... dr
        total_integral = cumulative_integral[-1]  # ∫_0^T
        integral_0_to_t = np.interp(t_np, r_grid, cumulative_integral)  # shape (batch,)
        integral_t_to_T = total_integral - integral_0_to_t  # shape (batch,)
        v_np = quadratic_term + integral_t_to_T  # shape (batch,)

        # transform to torch tensor, shape (batch, 1)
        v_torch = torch.tensor(v_np, dtype=x.dtype, device=x.device).unsqueeze(1)

        return v_torch

    # ================================================================
    # 3. Optimal a(t, x)
    # ================================================================
    def optimal_control(self, t, x):
        """
        Calculate the optimal Markov control: a(t, x) = -D^{-1} M^T S(t) x
        
        Parameters:
          t: torch.Tensor, shape (batch_size,)  Time for each sample
          x: torch.Tensor, shape (batch_size, 1, 2)  Spatial location for each sample (2D state)
        Returns:
          a: torch.Tensor, shape (batch_size, 2)  Optimal control for each (t, x) (2D control vector)
        
        Derivation: a(t, x) = -D^{-1} M^T S(t) x
        Where D^{-1} M^T is a (2,2) matrix, S(t) is a (2,2) matrix, and x is a (2,) vector
        Therefore, a(t,x) = -(D^{-1} M^T) @ S(t) @ x, the result is... (2,) vector
        """
        #Convert to NumPy
        t_np = t.detach().cpu().numpy()   # shape (batch,)
        x_np = x.detach().cpu().numpy()   # shape (batch, 1, 2)

        # Obtain the value of S(t) at each time point
        S_at_t = self._get_S_at_times(t_np)  # shape (batch, 2, 2)

        # x Remove the intermediate dimension: (batch, 1, 2) -> (batch, 2)
        x_2d = x_np[:, 0, :]  # shape (batch, 2)

        # Compute a = -D^{-1} M^⊤ S(t) x
        # D_inv_MT shape (2,2), S_at_t[i] shape (2,2), x_2d[i] shape (2,)
        # for each sample i: a_i = -D_inv_MT @ S_at_t[i] @ x_2d[i]
        # einsum: a[i,j] = -D_inv_MT[j,k] * S_at_t[i,k,l] * x_2d[i,l]
        a_np = -np.einsum('jk,ikl,il->ij', self.D_inv_MT, S_at_t, x_2d)  # shape (batch, 2)

        # convert torch tensor
        a_torch = torch.tensor(a_np, dtype=x.dtype, device=x.device)

        return a_torch


# ====================================================================
# Test / Example usage
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1.1: LQR Solver Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Define test parameters
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Create an LQR and solve Riccati ODE
    # ------------------------------------------------------------------
    lqr = LQR(H, M, sigma, C, D, R_mat, T)

    # Solving on a uniform time grid
    N_time = 1000
    time_grid = np.linspace(0, T, N_time + 1)
    S_values = lqr.solve_riccati(time_grid)

    print(f"\nRiccati ODE solved on {N_time + 1} time points.")
    print(f"S(0) = \n{S_values[0]}")
    print(f"S(T) = \n{S_values[-1]}")
    print(f"S(T) should equal R = \n{R_mat}")

    # Check if S(T) is equal to the terminal condition R
    assert np.allclose(S_values[-1], R_mat, atol=1e-10), \
        f"S(T) does not match R! Got {S_values[-1]}"
    print("S(T) matches terminal condition R")

    # ------------------------------------------------------------------
    # Test value function
    # ------------------------------------------------------------------
    batch_size = 5
    t_test = torch.rand(batch_size) * T          # random time
    x_test = torch.randn(batch_size, 1, 2)       # random state

    v = lqr.value_function(t_test, x_test)
    print(f"\nValue function test:")
    print(f"  t = {t_test.numpy()}")
    print(f"  x = {x_test.squeeze(1).numpy()}")
    print(f"  v(t,x) = {v.squeeze(1).numpy()}")
    print(f"  v shape = {v.shape}  (expected: ({batch_size}, 1))")

    # The value function should be non-negative 
    # because C, D, and R are all positive semi-definite
    assert (v >= -1e-8).all(), f"Value function has negative values: {v}"
    print("Value function is non-negative (as expected)")

    # ------------------------------------------------------------------
    # Test optimal control
    # ------------------------------------------------------------------
    a = lqr.optimal_control(t_test, x_test)
    print(f"\nOptimal control test:")
    print(f"  a(t,x) = {a.numpy()}")
    print(f"  a shape = {a.shape}  (expected: ({batch_size}, 2))")

    # ------------------------------------------------------------------
    # Verification: At t=T, S(T) = R = I, therefore a(T,x) = -D^{-1} M^⊤ R x = -x
    # when D=I, M=I, R=I
    # ------------------------------------------------------------------
    t_T = torch.ones(3) * T
    x_check = torch.tensor([[[1.0, 2.0]], [[3.0, -1.0]], [[-2.0, 0.5]]])

    a_at_T = lqr.optimal_control(t_T, x_check)
    expected_a = -x_check.squeeze(1)
    print(f"\nControl at t=T (should be -x when D=M=R=I):")
    print(f"  a(T,x)    = {a_at_T.numpy()}")
    print(f"  -x        = {expected_a.numpy()}")
    assert torch.allclose(a_at_T, expected_a, atol=1e-8), \
        f"a(T,x) doesn't match -x! Diff: {(a_at_T - expected_a).abs().max()}"
    print("✓ a(T,x) = -x verified (as expected for D=M=R=I)")

    # ------------------------------------------------------------------
    # Verification: v(T, x) = x^⊤ R x = |x|^2 when R=I
    # ------------------------------------------------------------------
    v_at_T = lqr.value_function(t_T, x_check)
    expected_v = (x_check.squeeze(1) ** 2).sum(dim=1, keepdim=True)
    print(f"\nValue at t=T (should be |x|^2 when R=I):")
    print(f"  v(T,x)    = {v_at_T.numpy().flatten()}")
    print(f"  |x|^2     = {expected_v.numpy().flatten()}")
    assert torch.allclose(v_at_T, expected_v, atol=1e-6), \
        f"v(T,x) doesn't match |x|^2! Diff: {(v_at_T - expected_v).abs().max()}"
    print("✓ v(T,x) = x^⊤ R x verified")

    print("\n" + "=" * 60)
    print("All Exercise 1.1 tests PASSED!")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Report Figure 1: Variation of elements of the S(t) matrix over time
    # Purpose: To show the solution of the Riccati ODE and verify the terminal condition S(T) = R   
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
 
    ax.plot(time_grid, S_values[:, 0, 0], 'b-',  linewidth=2, label='S(t)[0,0]')
    ax.plot(time_grid, S_values[:, 0, 1], 'r--', linewidth=2, label='S(t)[0,1]')
    ax.plot(time_grid, S_values[:, 1, 0], 'g-.', linewidth=2, label='S(t)[1,0]')
    ax.plot(time_grid, S_values[:, 1, 1], 'm:',  linewidth=2, label='S(t)[1,1]')
 
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('S(t) entries', fontsize=12)
    ax.set_title('Riccati ODE Solution: S(t) Matrix Entries over Time', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('fig1_riccati_solution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nfig1_riccati_solution.png has saved!")
 
    # ------------------------------------------------------------------
    # Figure 2: Heatmap of the value function v(0, x)
    # Purpose: To visually display the shape of the value function (should be bowl-shaped centred at the origin)
    # ------------------------------------------------------------------
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
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Value Function v(0, x) at t = 0', fontsize=13)
 
    plt.tight_layout()
    plt.savefig('fig2_value_function_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Exercise1.1 fig2_value_function_heatmap.png has saved!")
 