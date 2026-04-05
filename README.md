# SCDAA Coursework 2025-26: Policy Iteration with Deep Galerkin Method

## Group Information

| Name | Student Number | Contribution |
|------|---------------|-------------|
| [Finlay Charleson] | [s1908808] | 1/3 |
| [Huaying Wang] | [s2105829] | 1/3 |
| [Jinxiu Lu] | [s2884616] | 1/3 |

## Dependencies

- Python 3.10+
- numpy
- scipy
- matplotlib
- torch (PyTorch)

No other libraries are used, as required by the coursework specification.

## File Structure

```
├── exercise_1_1_lqr_solver.py        # Exercise 1.1: LQR class, Riccati ODE solver
├── exercise_1_2_lqr_mc.py            # Exercise 1.2: Monte Carlo verification
├── exercise_2_supervised_learning.py  # Exercise 2.1 & 2.2: Supervised learning of v and a
├── exercise_3_1_dgm.py               # Exercise 3.1: Deep Galerkin Method for linear PDE
├── exercise_4_1_pia_dgm.py           # Exercise 4.1: Policy Iteration with DGM
└── README.md
```

### Code Dependencies Between Files

```
exercise_1_1_lqr_solver.py          ← base module, no dependencies
exercise_1_2_lqr_mc.py              ← imports LQR from exercise_1_1
exercise_2_supervised_learning.py    ← imports LQR from exercise_1_1
exercise_3_1_dgm.py                 ← imports LQR from exercise_1_1
                                       imports NetDGM from exercise_2
exercise_4_1_pia_dgm.py             ← imports LQR from exercise_1_1
                                       imports NetDGM, FFN from exercise_2
                                       imports sample_interior, sample_terminal from exercise_3
```

## Quick to Reproduce All Results

### 1. Install Dependencies

```bash
pip install numpy scipy matplotlib torch
```
### 2. Run All Experiments

All figures and numerical results in the report can be reproduced by running:

```bash
python exercise_1_1_lqr_solver.py
python exercise_1_2_lqr_mc.py
python exercise_2_supervised_learning.py
python exercise_3_1_dgm.py
python exercise_4_1_pia_dgm.py
```

Generated outputs will be saved automatically in its corrosponding `experimenti` directory.

### Exercise 1.1 — Riccati ODE Solver

Solve the Riccati equation and obtain the analytical LQR solution.

Run
```bash
python exercise_1_1_lqr_solver.py
```

**Outputs:**
- `fig1_riccati_solution.png` — S(t) matrix entries over time
- `fig2_value_function_heatmap.png` — Value function v(0, x) contour plot

### Exercise 1.2 — Monte Carlo Verification

Verify numerical convergence using Monte-Carlo simulation(both implict and explict euler).

Run
```bash
python exercise_1_2_lqr_mc.py
```

**Outputs:**
- `fig3_time_discretisation_convergence.png` — Log-log plot: error vs time step size
- `fig4_mc_sample_convergence.png` — Log-log plot: error vs number of MC samples

**Note:** This takes approximately 5-10 minutes due to large MC simulations (up to 100,000 paths).

### Exercise 2.1 & 2.2 — Supervised Learning

Train neural networks to approximate the **value function** and the **optimal control policy** to verify the validation of using neural network to learn PDE solution and dynamic optimal strategy.

Run
```bash
python exercise_2_supervised_learning.py
```

**Outputs:**
- `exercise_2_1_value_loss.png` — Training loss for value function network
- `exercise_2_2_control_loss.png` — Training loss for control network
- `exercise_2_1_value_net.pt` — Trained value network weights
- `exercise_2_2_control_net.pt` — Trained control network weights
- `exercise_2_summary.txt` — Summary of test errors

### Exercise 3.1 — Deep Galerkin Method

Solve the linear PDE with a fixed control policy by the **Deep Galerkin Method**.

Run
```bash
python exercise_3_1_dgm.py
```

**Outputs:**
- `fig5_dgm_training_loss.png` — DGM training loss curve
- `fig6_dgm_mc_error.png` — DGM error against MC reference over training
- `exercise_3_1_validation.txt` — Point-by-point comparison table
- `exercise_3_1_dgm_weights.pt` — Trained network weights

### Exercise 4.1 — Policy Iteration with DGM

Implement **the Policy Iteration Algorithm (PIA)**, including policy evaluation and policy improvement in each iteration, with DGM network from exercise 3.

Run
```bash
python exercise_4_1_pia_dgm.py
```

**Outputs:**
- `fig7_pia_eval_loss.png` — Policy evaluation loss curve
- `fig8_pia_improve_loss.png` — Policy improvement Hamiltonian curve
- `fig9_pia_convergence.png` — Convergence of value and action errors to exact LQR
- `exercise_4_1_validation.txt` — Point-by-point comparison table
- `exercise_4_1_value_net.pt` — Trained value network weights
- `exercise_4_1_action_net.pt` — Trained action network weights

**Note:** This takes approximately 15-30 minutes depending on hardware. 

## Test Parameters

All exercises use the same set of parameters:

```
H = M = C = D = R = I  (2×2 identity matrix)
sigma = 0.5 * I
T = 1.0
```

Exercise 3 uses constant control alpha = (1, 1).

## Key Implementation Choices

- **Riccati ODE** (Exercise 1.1): Solved using `scipy.integrate.solve_ivp` with RK45, rtol=atol=1e-12, max_step=0.001, with `dense_output=True` for arbitrary-time interpolation.
- **MC simulation** (Exercise 1.2): Both explicit and implicit Euler schemes implemented, with chunked processing (batch size 20,000) to manage memory.
- **Network architectures** (Exercise 2): DGM network (NetDGM, width 100, depth 3) for value function; feed-forward network (FFN, 2 hidden layers, width 100) for control. Both reused in Exercises 3 and 4.
- **DGM training** (Exercise 3): 1000 epochs, batch size 1024, learning rate 1e-3, sampling domain [-3, 3]^2.
- **Policy iteration** (Exercise 4): 10 iterations, 500 evaluation epochs + 600 improvement epochs per iteration, batch size 1024, constant policy initialisation for action network, einsum-based PDE computation.