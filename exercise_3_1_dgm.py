import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class DGMNet(nn.Module):
    def __init__(self, in_dim=3, width=100, depth=3):
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


class LinearPDEProblem:
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
        t.requires_grad_(True)
        x.requires_grad_(True)
        u = net(t, x)
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


def sample_interior(batch_size, T, x_low=-3.0, x_high=3.0, device='cpu'):
    t = T * torch.rand(batch_size, 1, device=device)
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


def sample_terminal(batch_size, T, x_low=-3.0, x_high=3.0, device='cpu'):
    t = torch.full((batch_size, 1), T, device=device)
    x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=device)
    return t, x


def mc_value_constant_control(problem, t0, x0, n_steps=400, n_mc=4000, seed=1234):
    rng = np.random.default_rng(seed)
    tau = (problem.T - t0) / n_steps
    if tau <= 0:
        return float(x0 @ problem.R.numpy() @ x0)

    x = np.repeat(x0[None, :], n_mc, axis=0)
    running = np.zeros(n_mc)
    H = problem.H.numpy()
    M = problem.M.numpy()
    sigma = problem.sigma.numpy()
    C = problem.C.numpy()
    R = problem.R.numpy()
    alpha = problem.alpha.numpy().reshape(2)
    alpha_D_alpha = problem.alpha_D_alpha

    for _ in range(n_steps):
        running += tau * (np.einsum('bi,ij,bj->b', x, C, x) + alpha_D_alpha)
        dw = rng.standard_normal((n_mc, 2)) * np.sqrt(tau)
        x = x + tau * (x @ H.T + alpha @ M.T) + dw @ sigma.T

    terminal = np.einsum('bi,ij,bj->b', x, R, x)
    return float(np.mean(running + terminal))


def build_validation_set(problem):
    pts = [
        (0.0, np.array([0.0, 0.0])),
        (0.0, np.array([1.0, 0.5])),
        (0.0, np.array([-1.0, 1.0])),
        (0.25, np.array([0.5, -0.5])),
        (0.25, np.array([1.5, 0.0])),
        (0.5, np.array([0.0, 1.0])),
        (0.5, np.array([-1.0, -1.0])),
        (0.75, np.array([0.5, 0.5])),
        (0.75, np.array([-0.5, 1.5])),
    ]
    refs = []
    for i, (t0, x0) in enumerate(pts):
        n_steps = max(100, int((problem.T - t0) * 400))
        refs.append(mc_value_constant_control(problem, t0, x0, n_steps=n_steps, n_mc=5000, seed=2026 + i))
    return pts, np.array(refs)


def evaluate_mc_error(net, pts, refs, device='cpu'):
    net.eval()
    t = torch.tensor([[p[0]] for p in pts], dtype=torch.float32, device=device)
    x = torch.tensor(np.stack([p[1] for p in pts]), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = net(t, x).squeeze(1).cpu().numpy()
    denom = np.maximum(1.0, np.abs(refs))
    return float(np.mean(np.abs(pred - refs) / denom))


def train_dgm(problem, epochs=1000, batch_size=256, lr=1e-3, device='cpu'):
    net = DGMNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_history = []
    error_steps = []
    error_history = []
    pts, refs = build_validation_set(problem)

    for epoch in range(1, epochs + 1):
        net.train()
        t_in, x_in = sample_interior(batch_size, problem.T, device=device)
        t_bd, x_bd = sample_terminal(batch_size, problem.T, device=device)

        residual = problem.residual(net, t_in, x_in)
        boundary = net(t_bd, x_bd) - problem.terminal_value(x_bd)
        loss = torch.mean(residual ** 2) + torch.mean(boundary ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 100 == 0 or epoch == 1:
            err = evaluate_mc_error(net, pts, refs, device=device)
            error_steps.append(epoch)
            error_history.append(err)
            print(f"epoch={epoch:4d} loss={loss.item():.6e} mc_error={err:.6e}")

    return net, loss_history, error_steps, error_history, pts, refs


def plot_training(loss_history, error_steps, error_history):
    plt.figure(figsize=(7, 5))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.title('Exercise 3.1 DGM Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_3_1_loss.png', dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.semilogy(error_steps, error_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean relative error vs MC')
    plt.title('Exercise 3.1 Error Against Monte Carlo')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_3_1_mc_error.png', dpi=150)
    plt.close()


def save_predictions(net, pts, refs, device='cpu'):
    t = torch.tensor([[p[0]] for p in pts], dtype=torch.float32, device=device)
    x = torch.tensor(np.stack([p[1] for p in pts]), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = net(t, x).squeeze(1).cpu().numpy()

    with open('exercise_3_1_validation.txt', 'w', encoding='utf-8') as f:
        f.write('t,x1,x2,dgm_value,mc_value,rel_error\n')
        for (t0, x0), p, r in zip(pts, pred, refs):
            rel = abs(p - r) / max(1.0, abs(r))
            f.write(f'{t0:.2f},{x0[0]:.4f},{x0[1]:.4f},{p:.6f},{r:.6f},{rel:.6e}\n')


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    D = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    alpha = np.array([1.0, 1.0], dtype=np.float32)
    T = 1.0

    problem = LinearPDEProblem(H, M, sigma, C, D, R, T, alpha)
    net, loss_history, error_steps, error_history, pts, refs = train_dgm(
        problem,
        epochs=1000,
        batch_size=256,
        lr=1e-3,
        device=device,
    )

    plot_training(loss_history, error_steps, error_history)
    save_predictions(net, pts, refs, device=device)
    torch.save(net.state_dict(), 'exercise_3_1_dgm_weights.pt')
    print('Saved: exercise_3_1_loss.png, exercise_3_1_mc_error.png, exercise_3_1_validation.txt, exercise_3_1_dgm_weights.pt')
