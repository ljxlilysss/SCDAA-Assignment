import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class LQR:
    def __init__(self, H, M, sigma, C, D, R, T):
        self.H = np.array(H, dtype=np.float64)
        self.M = np.array(M, dtype=np.float64)
        self.sigma = np.array(sigma, dtype=np.float64)
        self.C = np.array(C, dtype=np.float64)
        self.D = np.array(D, dtype=np.float64)
        self.R = np.array(R, dtype=np.float64)
        self.T = float(T)
        self.D_inv = np.linalg.inv(self.D)
        self.D_inv_MT = self.D_inv @ self.M.T
        self.M_Dinv_MT = self.M @ self.D_inv_MT
        self.sigma_sigmaT = self.sigma @ self.sigma.T
        self._time_grid = None
        self._S_values = None
        self._const_values = None

    def solve_riccati(self, time_grid):
        time_grid = np.array(time_grid, dtype=np.float64)
        H_T = self.H.T
        M_Dinv_MT = self.M_Dinv_MT
        C = self.C

        def rhs(tau, y):
            S = y.reshape(2, 2)
            dS = 2.0 * H_T @ S - S @ M_Dinv_MT @ S + C
            return dS.ravel()

        tau_eval = np.sort(self.T - time_grid)
        sol = solve_ivp(rhs, (0.0, self.T), self.R.ravel(), t_eval=tau_eval, rtol=1e-10, atol=1e-10)
        if not sol.success:
            raise RuntimeError(sol.message)

        S_values = sol.y[:, ::-1].T.reshape(-1, 2, 2)
        const_values = np.zeros(len(time_grid))
        for i, t in enumerate(time_grid):
            grid = np.linspace(t, self.T, 300)
            S_grid = self._interp_mats(grid, time_grid, S_values)
            traces = np.einsum('ij,nji->n', self.sigma_sigmaT, S_grid)
            const_values[i] = np.trapz(traces, grid)

        self._time_grid = time_grid
        self._S_values = S_values
        self._const_values = const_values
        return S_values

    @staticmethod
    def _interp_mats(times, grid, mats):
        out = np.zeros((len(times), 2, 2))
        for i in range(2):
            for j in range(2):
                out[:, i, j] = np.interp(times, grid, mats[:, i, j])
        return out

    def _get_S(self, t):
        t = np.array(t, dtype=np.float64).reshape(-1)
        return self._interp_mats(t, self._time_grid, self._S_values)

    def _get_const(self, t):
        t = np.array(t, dtype=np.float64).reshape(-1)
        return np.interp(t, self._time_grid, self._const_values)

    def value_function(self, t, x):
        t_np = t.detach().cpu().numpy().reshape(-1)
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, 0, :]
        S = self._get_S(t_np)
        c = self._get_const(t_np)
        quad = np.einsum('ni,nij,nj->n', x_np, S, x_np)
        v = quad + c
        return torch.tensor(v[:, None], dtype=t.dtype, device=t.device)

    def optimal_control(self, t, x):
        t_np = t.detach().cpu().numpy().reshape(-1)
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, 0, :]
        S = self._get_S(t_np)
        a = -np.einsum('ij,njk,nk->ni', self.D_inv_MT, S, x_np)
        return torch.tensor(a, dtype=x.dtype, device=x.device)


class DGMLayer(nn.Module):
    def __init__(self, width, in_dim):
        super().__init__()
        self.z_x = nn.Linear(in_dim, width)
        self.z_h = nn.Linear(width, width)
        self.g_x = nn.Linear(in_dim, width)
        self.g_h = nn.Linear(width, width)
        self.r_x = nn.Linear(in_dim, width)
        self.r_h = nn.Linear(width, width)
        self.h_x = nn.Linear(in_dim, width)
        self.h_h = nn.Linear(width, width)

    def forward(self, x, h):
        z = torch.tanh(self.z_x(x) + self.z_h(h))
        g = torch.sigmoid(self.g_x(x) + self.g_h(h))
        r = torch.sigmoid(self.r_x(x) + self.r_h(h))
        h_tilde = torch.tanh(self.h_x(x) + self.h_h(r * h))
        return (1.0 - g) * h_tilde + z * h


class ValueNet(nn.Module):
    def __init__(self, in_dim=3, width=80, depth=3):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, width), nn.Tanh())
        self.layers = nn.ModuleList([DGMLayer(width, in_dim) for _ in range(depth)])
        self.output_layer = nn.Linear(width, 1)

    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        h = self.input_layer(inp)
        for layer in self.layers:
            h = layer(inp, h)
        return self.output_layer(h)


class ActionNet(nn.Module):
    def __init__(self, in_dim=3, width=64, depth=3, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))


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
        drift_x = torch.sum((x @ self.H.T) * grad_x, dim=1, keepdim=True)
        drift_a = torch.sum((a @ self.M.T) * grad_x, dim=1, keepdim=True)
        running_x = torch.sum((x @ self.C) * x, dim=1, keepdim=True)
        running_a = torch.sum((a @ self.D) * a, dim=1, keepdim=True)
        return u_t + diff_term + drift_x + drift_a + running_x + running_a

    def hamiltonian(self, value_net, action_net, t, x):
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)
        u = value_net(t, x)
        grad_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        a = action_net(t, x)
        drift_x = torch.sum((x @ self.H.T) * grad_x, dim=1, keepdim=True)
        drift_a = torch.sum((a @ self.M.T) * grad_x, dim=1, keepdim=True)
        running_x = torch.sum((x @ self.C) * x, dim=1, keepdim=True)
        running_a = torch.sum((a @ self.D) * a, dim=1, keepdim=True)
        return drift_x + drift_a + running_x + running_a


def sample_interior(batch_size, T, x_low=-2.0, x_high=2.0, device='cpu'):
    t = T * torch.rand(batch_size, 1, device=device)
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


def sample_terminal(batch_size, T, x_low=-2.0, x_high=2.0, device='cpu'):
    t = torch.full((batch_size, 1), T, device=device)
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


def evaluate_errors(value_net, action_net, exact, t_eval, x_eval, device='cpu'):
    value_net.eval()
    action_net.eval()
    with torch.no_grad():
        v_pred = value_net(t_eval, x_eval).cpu().numpy().squeeze()
        a_pred = action_net(t_eval, x_eval).cpu().numpy()
        v_true = exact.value_function(t_eval, x_eval).cpu().numpy().squeeze()
        a_true = exact.optimal_control(t_eval, x_eval).cpu().numpy()
    v_err = np.mean(np.abs(v_pred - v_true) / np.maximum(1.0, np.abs(v_true)))
    a_err = np.mean(np.linalg.norm(a_pred - a_true, axis=1) / np.maximum(1.0, np.linalg.norm(a_true, axis=1)))
    return float(v_err), float(a_err)


def train_policy_evaluation(problem, value_net, action_net, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
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


def run_pia(problem, exact, n_policy_iter=6, eval_epochs=400, improve_epochs=300, batch_size=256, lr_val=1e-3, lr_act=1e-3):
    value_net = ValueNet().to(problem.device)
    action_net = ActionNet().to(problem.device)

    t_eval = torch.rand(256, 1, device=problem.device) * problem.T
    x_eval = -2.0 + 4.0 * torch.rand(256, 2, device=problem.device)

    eval_loss_hist = []
    improve_loss_hist = []
    value_err_hist = []
    action_err_hist = []

    for k in range(n_policy_iter):
        eval_hist = train_policy_evaluation(problem, value_net, action_net, eval_epochs, batch_size, lr_val)
        improve_hist = train_policy_improvement(problem, value_net, action_net, improve_epochs, batch_size, lr_act)
        v_err, a_err = evaluate_errors(value_net, action_net, exact, t_eval, x_eval, problem.device)
        eval_loss_hist.extend(eval_hist)
        improve_loss_hist.extend(improve_hist)
        value_err_hist.append(v_err)
        action_err_hist.append(a_err)
        print(f'iter={k + 1} value_err={v_err:.6e} action_err={a_err:.6e}')

    return value_net, action_net, eval_loss_hist, improve_loss_hist, value_err_hist, action_err_hist, t_eval, x_eval


def save_plots(eval_loss_hist, improve_loss_hist, value_err_hist, action_err_hist):
    plt.figure(figsize=(7, 5))
    plt.semilogy(eval_loss_hist)
    plt.xlabel('Evaluation step')
    plt.ylabel('Loss')
    plt.title('Exercise 4.1 Policy Evaluation Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_4_1_eval_loss.png', dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(improve_loss_hist)
    plt.xlabel('Improvement step')
    plt.ylabel('Hamiltonian')
    plt.title('Exercise 4.1 Policy Improvement Objective')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_4_1_improve_loss.png', dpi=150)
    plt.close()

    xs = np.arange(1, len(value_err_hist) + 1)
    plt.figure(figsize=(7, 5))
    plt.semilogy(xs, value_err_hist, marker='o', label='Value error')
    plt.semilogy(xs, action_err_hist, marker='s', label='Action error')
    plt.xlabel('Policy iteration')
    plt.ylabel('Mean relative error')
    plt.title('Exercise 4.1 Convergence to Exact LQR')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('exercise_4_1_convergence.png', dpi=150)
    plt.close()


def save_validation(value_net, action_net, exact, device='cpu'):
    t = torch.tensor([[0.0], [0.0], [0.25], [0.5], [0.75]], dtype=torch.float32, device=device)
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5], [-1.0, 1.0], [1.5, 0.0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        v_pred = value_net(t, x).cpu().numpy().squeeze()
        a_pred = action_net(t, x).cpu().numpy()
        v_true = exact.value_function(t, x).cpu().numpy().squeeze()
        a_true = exact.optimal_control(t, x).cpu().numpy()
    with open('exercise_4_1_validation.txt', 'w', encoding='utf-8') as f:
        f.write('t,x1,x2,v_pred,v_true,v_rel_err,a1_pred,a2_pred,a1_true,a2_true,a_rel_err\n')
        for i in range(len(t)):
            v_rel = abs(v_pred[i] - v_true[i]) / max(1.0, abs(v_true[i]))
            a_rel = np.linalg.norm(a_pred[i] - a_true[i]) / max(1.0, np.linalg.norm(a_true[i]))
            f.write(
                f'{t[i,0].item():.2f},{x[i,0].item():.4f},{x[i,1].item():.4f},'
                f'{v_pred[i]:.6f},{v_true[i]:.6f},{v_rel:.6e},'
                f'{a_pred[i,0]:.6f},{a_pred[i,1]:.6f},{a_true[i,0]:.6f},{a_true[i,1]:.6f},{a_rel:.6e}\n'
            )


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    D = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    T = 1.0

    exact = LQR(H, M, sigma, C, D, R, T)
    exact.solve_riccati(np.linspace(0.0, T, 1001))

    problem = PIAProblem(H, M, sigma, C, D, R, T, device=device)
    value_net, action_net, eval_loss_hist, improve_loss_hist, value_err_hist, action_err_hist, _, _ = run_pia(
        problem,
        exact,
        n_policy_iter=6,
        eval_epochs=400,
        improve_epochs=300,
        batch_size=256,
        lr_val=1e-3,
        lr_act=1e-3,
    )

    save_plots(eval_loss_hist, improve_loss_hist, value_err_hist, action_err_hist)
    save_validation(value_net, action_net, exact, device=device)
    torch.save(value_net.state_dict(), 'exercise_4_1_value_net.pt')
    torch.save(action_net.state_dict(), 'exercise_4_1_action_net.pt')
    print('Saved: exercise_4_1_eval_loss.png, exercise_4_1_improve_loss.png, exercise_4_1_convergence.png, exercise_4_1_validation.txt, exercise_4_1_value_net.pt, exercise_4_1_action_net.pt')
