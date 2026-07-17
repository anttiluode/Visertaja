import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ---------- The Chirp Latent Cell ----------
class ChirpLatentCell(nn.Module):
    def __init__(self, latent_dim, dt=0.1, steps=5, max_freq=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = dt
        self.steps = steps
        self.max_freq = max_freq
        self.fc_r = nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_omega = nn.Linear(latent_dim, latent_dim, bias=True)
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
        dr = self.fc_r(x)
        domega = self.fc_omega(x)
        for _ in range(self.steps):
            r = r + self.dt * (r * (1 - r**2) + dr)
            r = torch.clamp(r, min=0.0)
            omega = omega + self.dt * domega
            omega = torch.clamp(omega, -self.max_freq, self.max_freq)
            phi = phi + self.dt * omega
        phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
        out = r * torch.cos(phi)
        return out, (r, phi, omega)

# ---------- MNIST Model ----------
class ChirpMNIST(nn.Module):
    def __init__(self, latent_dim=64, steps=5):
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
        self.cell = ChirpLatentCell(latent_dim, dt=0.1, steps=steps)
        self.fc_out = nn.Linear(latent_dim, 10)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc_in(h)
        out, state = self.cell(h)
        logits = self.fc_out(out)
        return logits, state

# ---------- Training ----------
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

# ---------- Main ----------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=128, shuffle=False)

    model = ChirpMNIST(latent_dim=64, steps=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        train_epoch(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Visualize one example
    visualize_chirp(model, device, test_loader)

def visualize_chirp(model, device, test_loader):
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:1].to(device)

    with torch.no_grad():
        _, state = model(data)   # initial state after full forward
        r, phi, omega = state

        # Now manually step for more steps
        h = model.conv(data)
        h = model.fc_in(h)
        dr = model.cell.fc_r(h)
        domega = model.cell.fc_omega(h)
        dt = model.cell.dt
        steps = 50
        r_vals, phi_vals, omega_vals, out_vals = [], [], [], []
        for _ in range(steps):
            r = r + dt * (r * (1 - r**2) + dr)
            r = torch.clamp(r, min=0.0)
            omega = omega + dt * domega
            omega = torch.clamp(omega, -model.cell.max_freq, model.cell.max_freq)
            phi = phi + dt * omega
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
            out = r * torch.cos(phi)
            r_vals.append(r[0, 0].item())
            phi_vals.append(phi[0, 0].item())
            omega_vals.append(omega[0, 0].item())
            out_vals.append(out[0, 0].item())

        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(r_vals); axs[0, 0].set_title("Amplitude")
        axs[0, 1].plot(np.unwrap(phi_vals)); axs[0, 1].set_title("Phase (unwrapped)")
        axs[1, 0].plot(omega_vals); axs[1, 0].set_title("Frequency")
        axs[1, 1].plot(out_vals); axs[1, 1].set_title("Output (real part)")
        plt.tight_layout()
        plt.savefig("chirp_trajectory.png")
        print("Saved chirp_trajectory.png")

if __name__ == '__main__':
    main()