"""
resonant_attention.py — does the phase structure of the chirping trajectory
carry class information, or is the wave decorative?

Three readouts, same conv trunk, same chirp cell, trained separately:

  snapshot : linear on the FINAL output r·cos(φ)          (the V2 status quo)
  blind    : linear on the TIME-MEAN of amplitude r only  (sees the trajectory,
             cannot see phase — the phase-blind control)
  resonant : chirp-matched filter bank. Each class owns, per unit, a template
             wave  U(t) = exp(-i(ν t + ½κ t² + θ))  with learnable frequency ν,
             chirp rate κ, phase θ. The logit is the weighted |correlation|
             between the unit's complex trajectory Z(t) = r e^{iφ} and the
             template — i.e. energy transfers only when the unit's chirp
             sweeps through and phase-locks the template. Attention = resonance.

Registered predictions (before running):
  R1: resonant ≥ snapshot. KILL: resonant < snapshot − 0.3% → the integral
      of the wave adds nothing over its endpoint.
  R2 (the load-bearing one): resonant > blind by ≥ 0.3%. KILL: blind matches
      resonant → amplitude alone explains it; interference is decorative.
  Note: on MNIST everything scores ~99%, so ties are expected and honest.
      The instrument matters more than tonight's score — it ports unchanged
      to sequential/temporal tasks where the hypothesis has room to breathe.

Usage:  python resonant_attention.py            (trains all three, 3 epochs each)
        python resonant_attention.py resonant   (just one arm)
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

STEPS = 16      # internal thinking window (trajectory length)
DT = 0.1
LATENT = 64


class ChirpCellTraj(nn.Module):
    """V2 feedback dynamics, but records the full complex trajectory."""
    def __init__(self, latent_dim=LATENT, dt=DT, steps=STEPS,
                 alpha=1.0, beta=0.5, gamma=0.1, max_freq=20.0):
        super().__init__()
        self.dt, self.steps, self.max_freq = dt, steps, max_freq
        self.fc_r = nn.Linear(latent_dim, latent_dim)
        self.fc_omega = nn.Linear(latent_dim, latent_dim)
        self.alpha = nn.Parameter(torch.ones(latent_dim) * alpha)
        self.beta = nn.Parameter(torch.ones(latent_dim) * beta)
        self.gamma = nn.Parameter(torch.ones(latent_dim) * gamma)
        self.init_r = nn.Parameter(torch.ones(latent_dim) * 0.5)
        self.init_phi = nn.Parameter(torch.zeros(latent_dim))
        self.init_omega = nn.Parameter(torch.randn(latent_dim) * 0.5)

    def forward(self, x):
        B = x.size(0)
        r = self.init_r.expand(B, -1).clone()
        phi = self.init_phi.expand(B, -1).clone()
        omega = self.init_omega.expand(B, -1).clone()
        r_drive = self.fc_r(x)
        omega_drive = self.fc_omega(x)
        traj_re, traj_im, traj_r = [], [], []
        for _ in range(self.steps):
            r = torch.clamp(r + self.dt * (r * (1 - r**2) + r_drive), min=0.0)
            domega = (omega_drive + self.alpha * (r - 1.0)
                      + self.beta * torch.sin(phi) - self.gamma * omega)
            omega = torch.clamp(omega + self.dt * domega,
                                -self.max_freq, self.max_freq)
            phi = phi + self.dt * omega
            traj_re.append(r * torch.cos(phi))
            traj_im.append(r * torch.sin(phi))
            traj_r.append(r)
        Zre = torch.stack(traj_re, dim=1)   # (B, T, D)
        Zim = torch.stack(traj_im, dim=1)
        R = torch.stack(traj_r, dim=1)
        return Zre, Zim, R


class ResonantHead(nn.Module):
    """Per-(class, unit) learnable chirp template; logit = weighted |corr|."""
    def __init__(self, latent_dim=LATENT, n_classes=10, steps=STEPS, dt=DT):
        super().__init__()
        self.nu = nn.Parameter(torch.randn(n_classes, latent_dim) * 2.0)
        self.kappa = nn.Parameter(torch.zeros(n_classes, latent_dim))
        self.theta = nn.Parameter(torch.zeros(n_classes, latent_dim))
        self.w = nn.Parameter(torch.randn(n_classes, latent_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.register_buffer('t', torch.arange(steps).float() * dt)

    def forward(self, Zre, Zim):
        t = self.t[None, :, None]                        # (1, T, 1)
        ph = (self.nu[:, None, :] * t
              + 0.5 * self.kappa[:, None, :] * t**2
              + self.theta[:, None, :])                  # (C, T, D)
        Ure, Uim = torch.cos(ph), torch.sin(ph)
        # corr_c,d = (1/T) Σ_t Z conj(U):  Re = Zre·Ure + Zim·Uim,
        #                                  Im = Zim·Ure − Zre·Uim
        cre = torch.einsum('btd,ctd->bcd', Zre, Ure) \
            + torch.einsum('btd,ctd->bcd', Zim, Uim)
        cim = torch.einsum('btd,ctd->bcd', Zim, Ure) \
            - torch.einsum('btd,ctd->bcd', Zre, Uim)
        mag = torch.sqrt(cre**2 + cim**2 + 1e-8) / Zre.size(1)
        return (mag * self.w[None]).sum(-1) + self.bias


class ChirpNet(nn.Module):
    def __init__(self, readout='resonant'):
        super().__init__()
        self.readout = readout
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.fc_in = nn.Linear(32 * 7 * 7, LATENT)
        self.cell = ChirpCellTraj()
        if readout == 'resonant':
            self.head = ResonantHead()
        else:                       # snapshot and blind: plain linear heads
            self.head = nn.Linear(LATENT, 10)

    def forward(self, x):
        Zre, Zim, R = self.cell(self.fc_in(self.conv(x)))
        if self.readout == 'resonant':
            return self.head(Zre, Zim)
        if self.readout == 'blind':
            return self.head(R.mean(dim=1))          # time-mean amplitude only
        return self.head(Zre[:, -1, :])              # snapshot: final output


def run_arm(name, device, train_loader, test_loader, epochs=3):
    torch.manual_seed(0)
    model = ChirpNet(readout=name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(1) == target).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    print(f"  {name:9s}  {acc:6.2f}%   ({n_params:,} params)")
    return acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True,
                              download=True, transform=tfm),
                              batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False,
                             transform=tfm), batch_size=256, shuffle=False)

    arms = [sys.argv[1]] if len(sys.argv) > 1 else \
           ['snapshot', 'blind', 'resonant']
    print(f"\n=== Resonant attention, {STEPS}-step window, "
          f"3 epochs per arm ===")
    results = {a: run_arm(a, device, train_loader, test_loader) for a in arms}

    if len(results) == 3:
        print("\nLedger verdicts:")
        d1 = results['resonant'] - results['snapshot']
        d2 = results['resonant'] - results['blind']
        print(f"  R1 (resonant vs snapshot): {d1:+.2f}%  "
              f"{'[V] wave >= endpoint' if d1 >= -0.3 else '[K] KILLED'}")
        print(f"  R2 (resonant vs phase-blind): {d2:+.2f}%  "
              f"{'[V] interference load-bearing' if d2 >= 0.3 else '[~] tie — phase decorative on MNIST, port to temporal task'}")


if __name__ == '__main__':
    main()
