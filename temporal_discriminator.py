"""
temporal_discriminator.py — the registered coin-flip.

Task: row-sequential MNIST. The image arrives as a STREAM of 28 rows
(28 pixels each). No conv trunk, no peeking at the whole image: whatever
the model knows at the end, it accumulated through its recurrent state.
This is the first task where "representation = trajectory" can beat
"representation = vector" or die honestly.

Three arms, seeds fixed, gradient-clipped identically:

  gru             : nn.GRU, hidden size chosen by exact parameter search
                    (see below), linear readout on final hidden state.
  chirp_snapshot  : ChirpRNN core, linear readout on final r·cos(φ).
  chirp_resonant  : ChirpRNN core, chirp-template resonant readout over the
                    28-row output trajectory.

The chirp recurrent core: state (r, φ, ω) persists across rows — the
oscillator IS the memory. Each row's input, concatenated with the previous
output wave [r·cosφ, r·sinφ], drives amplitude and frequency; the V2
feedback terms α(r−1) + β·sin(φ) − γω keep the clock self-modulating.
History is stored in phase: two different row-orders leave the oscillator
at different points of its cycle even if the final drives are identical.
That is the thing a GRU must spend gate parameters to do and an oscillator
gets from its clock for free — IF it can be trained to use it.

PARAMETER MATCHING — where these comparisons get quietly rigged, so done
in the open: we count the chirp arm's exact trainable parameters, then
search GRU hidden sizes and pick the SMALLEST H whose total is >= the
chirp total (ties broken toward the GRU: it gets at least as many
parameters, never fewer). Both counts are printed. A chirp win cannot be
attributed to capacity; a chirp loss is a clean kill.

Registered before running:
  T1 (the coin-flip, prior undeclared): chirp_resonant vs GRU.
      KILL for the trajectory hypothesis: chirp_resonant < GRU − 1.0%
      (echoing the GeometricNeuron/GRU verdict — the ledger survives
      either way, the hypothesis may not).
  T2: chirp_resonant > chirp_snapshot by >= 0.3% → episode integration
      earns its keep on temporal input (it tied on static MNIST; here it
      has an actual stream to integrate). KILL: snapshot >= resonant.
  Guard: any arm failing to beat 90% is a training failure, not a verdict
      — rerun with lr 3e-4 before believing anything.

Runtime: ~5–10 min/epoch on CPU for the chirp arms (28 rows × 3 internal
steps). 3 epochs per arm. Usage:
  python temporal_discriminator.py             (all three arms)
  python temporal_discriminator.py gru         (one arm)
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

D = 64            # chirp latent units
ROWS = 28         # sequence length (rows of the image)
IN_DIM = 28       # pixels per row
INNER = 3         # internal oscillator steps per row
DT = 0.1
EPOCHS = 3
GRAD_CLIP = 1.0


# ---------------- Chirp recurrent core ----------------
class ChirpRNNCell(nn.Module):
    def __init__(self, in_dim=IN_DIM, latent=D, dt=DT, inner=INNER,
                 alpha=1.0, beta=0.5, gamma=0.1, max_freq=20.0):
        super().__init__()
        self.dt, self.inner, self.max_freq = dt, inner, max_freq
        drive_in = in_dim + 2 * latent          # row + [r·cosφ, r·sinφ]
        self.fc_r = nn.Linear(drive_in, latent)
        self.fc_omega = nn.Linear(drive_in, latent)
        self.alpha = nn.Parameter(torch.ones(latent) * alpha)
        self.beta = nn.Parameter(torch.ones(latent) * beta)
        self.gamma = nn.Parameter(torch.ones(latent) * gamma)
        self.init_r = nn.Parameter(torch.ones(latent) * 0.5)
        self.init_phi = nn.Parameter(torch.zeros(latent))
        self.init_omega = nn.Parameter(torch.randn(latent) * 0.5)

    def init_state(self, B):
        return (self.init_r.expand(B, -1).clone(),
                self.init_phi.expand(B, -1).clone(),
                self.init_omega.expand(B, -1).clone())

    def forward(self, x_row, state):
        r, phi, omega = state
        feats = torch.cat([x_row, r * torch.cos(phi), r * torch.sin(phi)],
                          dim=-1)
        r_drive = self.fc_r(feats)
        omega_drive = self.fc_omega(feats)
        for _ in range(self.inner):
            r = torch.clamp(r + self.dt * (r * (1 - r**2) + r_drive),
                            min=0.0)
            domega = (omega_drive + self.alpha * (r - 1.0)
                      + self.beta * torch.sin(phi) - self.gamma * omega)
            omega = torch.clamp(omega + self.dt * domega,
                                -self.max_freq, self.max_freq)
            phi = phi + self.dt * omega
        return (r, phi, omega)


class ResonantHead(nn.Module):
    """Chirp-template matched filters over the ROWS-long output trajectory."""
    def __init__(self, latent=D, n_classes=10, T=ROWS, dt=DT * INNER):
        super().__init__()
        self.nu = nn.Parameter(torch.randn(n_classes, latent) * 2.0)
        self.kappa = nn.Parameter(torch.zeros(n_classes, latent))
        self.theta = nn.Parameter(torch.zeros(n_classes, latent))
        self.w = nn.Parameter(torch.randn(n_classes, latent) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.register_buffer('t', torch.arange(T).float() * dt)

    def forward(self, Zre, Zim):                     # (B, T, D)
        t = self.t[None, :, None]
        ph = (self.nu[:, None, :] * t
              + 0.5 * self.kappa[:, None, :] * t**2
              + self.theta[:, None, :])
        Ure, Uim = torch.cos(ph), torch.sin(ph)
        cre = torch.einsum('btd,ctd->bcd', Zre, Ure) \
            + torch.einsum('btd,ctd->bcd', Zim, Uim)
        cim = torch.einsum('btd,ctd->bcd', Zim, Ure) \
            - torch.einsum('btd,ctd->bcd', Zre, Uim)
        mag = torch.sqrt(cre**2 + cim**2 + 1e-8) / Zre.size(1)
        return (mag * self.w[None]).sum(-1) + self.bias


class ChirpSeqNet(nn.Module):
    def __init__(self, readout='resonant'):
        super().__init__()
        self.readout = readout
        self.cell = ChirpRNNCell()
        self.head = ResonantHead() if readout == 'resonant' \
            else nn.Linear(D, 10)

    def forward(self, x):                            # x: (B, 1, 28, 28)
        rows = x.squeeze(1)                          # (B, 28 rows, 28 px)
        state = self.cell.init_state(rows.size(0))
        Zre, Zim = [], []
        for i in range(ROWS):
            state = self.cell(rows[:, i, :], state)
            r, phi, _ = state
            Zre.append(r * torch.cos(phi))
            Zim.append(r * torch.sin(phi))
        if self.readout == 'resonant':
            return self.head(torch.stack(Zre, 1), torch.stack(Zim, 1))
        return self.head(Zre[-1])                    # snapshot: final wave


class GRUNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(IN_DIM, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 10)

    def forward(self, x):
        _, h = self.gru(x.squeeze(1))
        return self.head(h[-1])


# ---------------- Open parameter matching ----------------
def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_gru(target):
    """Smallest hidden size whose GRU total >= target (GRU never smaller)."""
    for H in range(8, 512):
        if count(GRUNet(H)) >= target:
            return H
    raise RuntimeError("no match found")


# ---------------- Harness ----------------
def run_arm(name, model, device, train_loader, test_loader):
    torch.manual_seed(0)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for k, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if k % 100 == 0:
                print(f"    {name} ep{ep+1} [{k*len(data)}/60000] "
                      f"loss {loss.item():.4f}")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(1) == target).sum().item()
    return 100. * correct / len(test_loader.dataset)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True,
                              download=True, transform=tfm),
                              batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False,
                             transform=tfm), batch_size=256, shuffle=False)

    chirp_res = ChirpSeqNet('resonant')
    chirp_snap = ChirpSeqNet('snapshot')
    target = count(chirp_res)
    H = match_gru(target)
    gru = GRUNet(H)

    print("=== Temporal discriminator: row-sequential MNIST ===")
    print(f"  chirp_resonant params: {target:,}")
    print(f"  chirp_snapshot params: {count(chirp_snap):,}")
    print(f"  GRU hidden={H}, params: {count(gru):,} "
          f"(matched open: smallest H with >= chirp params)")

    arms = {'gru': gru, 'chirp_snapshot': chirp_snap,
            'chirp_resonant': chirp_res}
    if len(sys.argv) > 1:
        arms = {sys.argv[1]: arms[sys.argv[1]]}

    results = {}
    for name, model in arms.items():
        print(f"\n--- {name} ---")
        results[name] = run_arm(name, model, device,
                                train_loader, test_loader)
        print(f"  {name}: {results[name]:.2f}%")

    if len(results) == 3:
        print("\n=== Ledger verdicts ===")
        for k, v in results.items():
            print(f"  {k:16s} {v:6.2f}%")
        t1 = results['chirp_resonant'] - results['gru']
        t2 = results['chirp_resonant'] - results['chirp_snapshot']
        guard = min(results.values()) < 90.0
        if guard:
            print("  GUARD TRIPPED: an arm is under 90% — training "
                  "failure, rerun with lr=3e-4 before believing verdicts.")
        print(f"  T1 (chirp vs GRU):      {t1:+.2f}%  "
              f"{'[K] hypothesis KILLED (loses by >1%)' if t1 < -1.0 else '[V] chirp survives the GRU' if t1 > 0 else '[~] within noise of the GRU'}")
        print(f"  T2 (resonant vs snap):  {t2:+.2f}%  "
              f"{'[V] episode integration earns its keep' if t2 >= 0.3 else '[K] snapshot suffices even on a stream' if t2 < 0 else '[~] tie'}")


if __name__ == '__main__':
    main()
