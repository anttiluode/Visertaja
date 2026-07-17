import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class ChirpLatentCellV2(nn.Module):
    """
    A latent oscillator with internal feedback:
      dr/dt   = r*(1 - r^2) + r_drive
      dω/dt   = ω_drive + α*(r - 1) + β*sin(φ) - γ*ω
      dφ/dt   = ω

    The frequency now depends on the oscillator's own amplitude and phase,
    producing genuine chirping.
    """
    def __init__(self, latent_dim, dt=0.1, steps=8,
                 alpha=1.0, beta=0.5, gamma=0.1,
                 max_freq=20.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = dt
        self.steps = steps
        self.max_freq = max_freq

        # Learnable projections from input to (r_drive, ω_drive)
        self.fc_r = nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_omega = nn.Linear(latent_dim, latent_dim, bias=True)

        # Learnable feedback strengths (per unit)
        self.alpha = nn.Parameter(torch.ones(latent_dim) * alpha)
        self.beta  = nn.Parameter(torch.ones(latent_dim) * beta)
        self.gamma = nn.Parameter(torch.ones(latent_dim) * gamma)

        # Initial state parameters
        self.init_r = nn.Parameter(torch.ones(latent_dim) * 0.5)
        self.init_phi = nn.Parameter(torch.zeros(latent_dim))
        self.init_omega = nn.Parameter(torch.randn(latent_dim) * 0.5)

    def forward(self, x, state=None):
        batch_size = x.size(0)
        if state is None:
            r = self.init_r.unsqueeze(0).expand(batch_size, -1).clone()
            phi = self.init_phi.unsqueeze(0).expand(batch_size, -1).clone()
            omega = self.init_omega.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            r, phi, omega = state

        # Input drives
        r_drive = self.fc_r(x)
        omega_drive = self.fc_omega(x)

        # Integrate for 'steps' internal iterations
        for _ in range(self.steps):
            # Amplitude: Stuart-Landau
            r = r + self.dt * (r * (1 - r**2) + r_drive)
            r = torch.clamp(r, min=0.0)

            # Frequency update with feedback from (r, phi)
            # dω/dt = ω_drive + α*(r-1) + β*sin(φ) - γ*ω
            domega = (omega_drive
                      + self.alpha * (r - 1.0)
                      + self.beta * torch.sin(phi)
                      - self.gamma * omega)
            omega = omega + self.dt * domega
            # Keep within bounds to avoid explosion
            omega = torch.clamp(omega, -self.max_freq, self.max_freq)

            # Phase advance
            phi = phi + self.dt * omega

        # Normalize phase
        phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi

        # Output: real part
        out = r * torch.cos(phi)
        return out, (r, phi, omega)

# ---------- Model (unchanged) ----------
class ChirpMNIST(nn.Module):
    def __init__(self, latent_dim=64, steps=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc_in = nn.Linear(32 * 7 * 7, latent_dim)
        self.cell = ChirpLatentCellV2(latent_dim, dt=0.1, steps=steps,
                                      alpha=1.0, beta=0.5, gamma=0.1)
        self.fc_out = nn.Linear(latent_dim, 10)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc_in(h)
        out, state = self.cell(h)
        logits = self.fc_out(out)
        return logits, state

# ---------- Training and visualization (same as before) ----------
def train_epoch(model, device, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] Loss: {loss.item():.4f}')
    return total_loss / len(loader)

def test(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
    acc = 100. * correct / len(loader.dataset)
    print(f'Test accuracy: {acc:.2f}%')
    return acc

def visualize_chirp(model, device, test_loader):
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:1].to(device)

    with torch.no_grad():
        # Get the initial state after a full forward pass
        _, state = model(data)
        r, phi, omega = state

        # Recompute input drive for manual stepping
        h = model.conv(data)
        h = model.fc_in(h)
        r_drive = model.cell.fc_r(h)
        omega_drive = model.cell.fc_omega(h)
        alpha = model.cell.alpha
        beta = model.cell.beta
        gamma = model.cell.gamma
        dt = model.cell.dt
        steps = 80  # longer to see chirp

        r_vals, phi_vals, omega_vals, out_vals = [], [], [], []
        for _ in range(steps):
            # Amplitude
            r = r + dt * (r * (1 - r**2) + r_drive)
            r = torch.clamp(r, min=0.0)
            # Frequency
            domega = omega_drive + alpha * (r - 1.0) + beta * torch.sin(phi) - gamma * omega
            omega = omega + dt * domega
            omega = torch.clamp(omega, -model.cell.max_freq, model.cell.max_freq)
            # Phase
            phi = phi + dt * omega
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
            out = r * torch.cos(phi)

            r_vals.append(r[0, 0].item())
            phi_vals.append(phi[0, 0].item())
            omega_vals.append(omega[0, 0].item())
            out_vals.append(out[0, 0].item())

        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(r_vals)
        axs[0, 0].set_title("Amplitude")
        axs[0, 1].plot(np.unwrap(phi_vals))
        axs[0, 1].set_title("Phase (unwrapped)")
        axs[1, 0].plot(omega_vals)
        axs[1, 0].set_title("Frequency")
        axs[1, 1].plot(out_vals)
        axs[1, 1].set_title("Output (real part)")
        plt.tight_layout()
        plt.savefig("chirp_v2_trajectory.png")
        print("Saved chirp_v2_trajectory.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=128, shuffle=False)

    model = ChirpMNIST(latent_dim=64, steps=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        train_epoch(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    visualize_chirp(model, device, test_loader)

if __name__ == '__main__':
    main()