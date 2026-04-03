import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from exercise_1_1_lqr_solver import LQR



# Utilities


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Networks required for Q2


class DGMLayer(nn.Module):
    def __init__(self, width: int, in_dim: int):
        super().__init__()
        self.z_x = nn.Linear(in_dim, width)
        self.z_h = nn.Linear(width, width)
        self.g_x = nn.Linear(in_dim, width)
        self.g_h = nn.Linear(width, width)
        self.r_x = nn.Linear(in_dim, width)
        self.r_h = nn.Linear(width, width)
        self.h_x = nn.Linear(in_dim, width)
        self.h_h = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.z_x(x) + self.z_h(h))
        g = torch.sigmoid(self.g_x(x) + self.g_h(h))
        r = torch.sigmoid(self.r_x(x) + self.r_h(h))
        h_tilde = torch.tanh(self.h_x(x) + self.h_h(r * h))
        return (1.0 - g) * h_tilde + z * h


class NetDGM(nn.Module):
    """
    DGM-style network for scalar value function v(t, x).
    Input dimension = 3 for (t, x1, x2), output dimension = 1.
    Hidden width fixed to 100 to match the question.
    """
    def __init__(self, in_dim: int = 3, width: int = 100, depth: int = 3):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, width), nn.Tanh())
        self.layers = nn.ModuleList([DGMLayer(width, in_dim) for _ in range(depth)])
        self.output_layer = nn.Linear(width, 1)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(tx)
        for layer in self.layers:
            h = layer(tx, h)
        return self.output_layer(h)


class FFN(nn.Module):
    """
    Standard feedforward network for the 2D Markov control a(t, x).
    2 hidden layers of size 100 as requested.
    Input dimension = 3 for (t, x1, x2), output dimension = 2.
    """
    def __init__(self, in_dim: int = 3, hidden_dim: int = 100, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        return self.net(tx)



# Fast exact labels from Exercise 1.1


class ExactLQRLabels:
    """
    Fast label generator using the solved Riccati system from Exercise 1.1.
    This avoids repeatedly calling value_function() from the uploaded Q1 solver,
    which recomputes a large quadrature every call and is too slow for NN datasets.
    """
    def __init__(self, lqr: LQR, integral_grid_size: int = 4001):
        if not hasattr(lqr, '_sol_dense'):
            raise RuntimeError('You must call lqr.solve_riccati(...) before creating ExactLQRLabels.')

        self.lqr = lqr
        self.T = float(lqr.T)

        r_grid = np.linspace(0.0, self.T, integral_grid_size)
        S_grid = lqr._get_S_at_times(r_grid)
        trace_vals = np.einsum('ij,nij->n', lqr.sigma_sigmaT, S_grid)

        dr = r_grid[1] - r_grid[0]
        cumulative = np.zeros_like(r_grid)
        cumulative[1:] = np.cumsum(0.5 * (trace_vals[:-1] + trace_vals[1:]) * dr)

        self.r_grid = r_grid
        self.cumulative = cumulative
        self.total_integral = cumulative[-1]

    def value_and_control(self, t_np: np.ndarray, x_np: np.ndarray):
        """
        Parameters
        ----------
        t_np : shape (N,)
        x_np : shape (N,2)

        Returns
        -------
        v_np : shape (N,1)
        a_np : shape (N,2)
        """
        S_t = self.lqr._get_S_at_times(t_np)
        quad = np.einsum('ni,nij,nj->n', x_np, S_t, x_np)
        int_0_t = np.interp(t_np, self.r_grid, self.cumulative)
        int_t_T = self.total_integral - int_0_t
        v_np = (quad + int_t_T)[:, None]
        a_np = -np.einsum('ij,njk,nk->ni', self.lqr.D_inv_MT, S_t, x_np)
        return v_np.astype(np.float32), a_np.astype(np.float32)



# Data generation


def sample_tx(n_samples: int, T: float, x_low: float = -3.0, x_high: float = 3.0):
    t = np.random.uniform(0.0, T, size=(n_samples, 1)).astype(np.float32)
    x = np.random.uniform(x_low, x_high, size=(n_samples, 2)).astype(np.float32)
    tx = np.concatenate([t, x], axis=1)
    return tx


def build_supervised_dataset(exact: ExactLQRLabels, n_samples: int):
    tx = sample_tx(n_samples, exact.T)
    t_np = tx[:, 0]
    x_np = tx[:, 1:]
    v_np, a_np = exact.value_and_control(t_np, x_np)
    return (
        torch.tensor(tx, dtype=torch.float32),
        torch.tensor(v_np, dtype=torch.float32),
        torch.tensor(a_np, dtype=torch.float32),
    )



# Training and evaluation

def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = torch.norm(pred - target)
    den = torch.norm(target) + 1e-12
    return (num / den).item()


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    mse_sum = 0.0
    n = 0
    preds_all = []
    target_all = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        mse_sum += torch.sum((pred - yb) ** 2).item()
        n += yb.numel()
        preds_all.append(pred.cpu())
        target_all.append(yb.cpu())
    preds = torch.cat(preds_all, dim=0)
    targets = torch.cat(target_all, dim=0)
    mse = mse_sum / n
    rel = relative_l2(preds, targets)
    return mse, rel, preds, targets


def train_supervised_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rel_l2': [],
    }

    best_state = None
    best_val = np.inf

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += torch.sum((pred - yb) ** 2).item()
            n_train += yb.numel()

        train_loss = running_loss / n_train
        val_loss, val_rel, _, _ = evaluate_model(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rel_l2'].append(val_rel)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch:4d} train_mse={train_loss:.6e} "
                f"val_mse={val_loss:.6e} val_relL2={val_rel:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history



# Plotting and reporting


def plot_loss(history: dict, title: str, out_file: str):
    plt.figure(figsize=(7, 5))
    plt.semilogy(history['train_loss'], label='Train MSE')
    plt.semilogy(history['val_loss'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


@torch.no_grad()
def save_predictions_table(model: nn.Module, loader: DataLoader, device: str, out_file: str):
    model.eval()
    with open(out_file, 'w', encoding='utf-8') as f:
        first = True
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            xb_np = xb.cpu().numpy()
            yb_np = yb.cpu().numpy()

            if first:
                if pred.shape[1] == 1:
                    f.write('t,x1,x2,pred,true,abs_err\n')
                else:
                    f.write('t,x1,x2,pred1,pred2,true1,true2,l2_err\n')
                first = False

            for row_x, row_p, row_y in zip(xb_np, pred, yb_np):
                t, x1, x2 = row_x
                if pred.shape[1] == 1:
                    err = abs(row_p[0] - row_y[0])
                    f.write(f'{t:.6f},{x1:.6f},{x2:.6f},{row_p[0]:.8f},{row_y[0]:.8f},{err:.8e}\n')
                else:
                    err = np.linalg.norm(row_p - row_y)
                    f.write(
                        f'{t:.6f},{x1:.6f},{x2:.6f},'
                        f'{row_p[0]:.8f},{row_p[1]:.8f},'
                        f'{row_y[0]:.8f},{row_y[1]:.8f},{err:.8e}\n'
                    )


# ============================================================
# Main script
# ============================================================

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Same baseline matrices as your friends' uploaded files.
    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    D = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    T = 1.0

    # Step 1: Solve Exercise 1.1 exactly.
    lqr = LQR(H, M, sigma, C, D, R_mat, T)
    riccati_grid = np.linspace(0.0, T, 1001)
    lqr.solve_riccati(riccati_grid)
    exact = ExactLQRLabels(lqr, integral_grid_size=4001)

    # Step 2: Build one dataset shared by both Q2.1 and Q2.2.
    # This keeps everything consistent for later parts.
    n_train = 40000
    n_val = 8000
    n_test = 8000

    x_train, v_train, a_train = build_supervised_dataset(exact, n_train)
    x_val, v_val, a_val = build_supervised_dataset(exact, n_val)
    x_test, v_test, a_test = build_supervised_dataset(exact, n_test)

    batch_size = 512
    value_train_loader = DataLoader(TensorDataset(x_train, v_train), batch_size=batch_size, shuffle=True)
    value_val_loader = DataLoader(TensorDataset(x_val, v_val), batch_size=batch_size, shuffle=False)
    value_test_loader = DataLoader(TensorDataset(x_test, v_test), batch_size=batch_size, shuffle=False)

    control_train_loader = DataLoader(TensorDataset(x_train, a_train), batch_size=batch_size, shuffle=True)
    control_val_loader = DataLoader(TensorDataset(x_val, a_val), batch_size=batch_size, shuffle=False)
    control_test_loader = DataLoader(TensorDataset(x_test, a_test), batch_size=batch_size, shuffle=False)

    
    # Exercise 2.1: supervised learning of value function v
    
    print('\n' + '=' * 70)
    print('Exercise 2.1: Training DGM network for value function v(t,x)')
    print('=' * 70)

    value_net = NetDGM(in_dim=3, width=100, depth=3).to(device)
    value_net, value_hist = train_supervised_model(
        value_net,
        value_train_loader,
        value_val_loader,
        device=device,
        epochs=300,
        lr=1e-3,
        weight_decay=1e-6,
    )

    value_test_mse, value_test_rel, _, _ = evaluate_model(value_net, value_test_loader, device)
    print(f'Q2.1 TEST: mse={value_test_mse:.6e}, relL2={value_test_rel:.6e}')

    plot_loss(value_hist, 'Exercise 2.1 Training Loss: Value Function', 'exercise_2_1_value_loss.png')
    save_predictions_table(value_net, value_test_loader, device, 'exercise_2_1_value_test_predictions.csv')
    torch.save(value_net.state_dict(), 'exercise_2_1_value_net.pt')

   
    # Exercise 2.2: supervised learning of control a
   
    print('\n' + '=' * 70)
    print('Exercise 2.2: Training FFN for Markov control a(t,x)')
    print('=' * 70)

    control_net = FFN(in_dim=3, hidden_dim=100, out_dim=2).to(device)
    control_net, control_hist = train_supervised_model(
        control_net,
        control_train_loader,
        control_val_loader,
        device=device,
        epochs=300,
        lr=1e-3,
        weight_decay=1e-6,
    )

    control_test_mse, control_test_rel, _, _ = evaluate_model(control_net, control_test_loader, device)
    print(f'Q2.2 TEST: mse={control_test_mse:.6e}, relL2={control_test_rel:.6e}')

    plot_loss(control_hist, 'Exercise 2.2 Training Loss: Markov Control', 'exercise_2_2_control_loss.png')
    save_predictions_table(control_net, control_test_loader, device, 'exercise_2_2_control_test_predictions.csv')
    torch.save(control_net.state_dict(), 'exercise_2_2_control_net.pt')

    # Save one compact summary file for write-up / later parts.
    with open('exercise_2_summary.txt', 'w', encoding='utf-8') as f:
        f.write('Exercise 2 summary\n')
        f.write('===================\n')
        f.write(f'Device: {device}\n')
        f.write(f'Train/Val/Test sizes: {n_train}/{n_val}/{n_test}\n')
        f.write(f'Q2.1 value test MSE: {value_test_mse:.8e}\n')
        f.write(f'Q2.1 value test relL2: {value_test_rel:.8e}\n')
        f.write(f'Q2.2 control test MSE: {control_test_mse:.8e}\n')
        f.write(f'Q2.2 control test relL2: {control_test_rel:.8e}\n')

    print('\nSaved files:')
    print('  exercise_2_1_value_loss.png')
    print('  exercise_2_2_control_loss.png')
    print('  exercise_2_1_value_test_predictions.csv')
    print('  exercise_2_2_control_test_predictions.csv')
    print('  exercise_2_1_value_net.pt')
    print('  exercise_2_2_control_net.pt')
    print('  exercise_2_summary.txt')


if __name__ == '__main__':
    main()
