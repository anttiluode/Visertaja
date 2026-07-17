"""
inspect_population.py — the survey instrument for the [~] OPEN ledger item.

The shipped V2 trajectory plot showed unit [0,0] with amplitude 0: a silent
oscillator. This script answers the population question:

  * How many of the latent units are LIVE (r > 0.1 at end of rollout)?
  * Of the live ones, how many CHIRP (|mean dω/dt| above threshold)?
  * Distribution of chirp rates across units.

Echo of Nollas E2: if a frozen majority carries nothing and a mobile tail
carries everything, that is the two-speed medium again — report it either way.

Usage:
  Train first (chirp_v2_feedback.py saves nothing by default), so this script
  trains a fresh model quickly (5 epochs) then surveys, OR pass a saved
  state_dict path as argv[1] if you add torch.save to your training run.
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from chirp_v2_feedback import ChirpMNIST, train_epoch, test
import torch.optim as optim

ROLLOUT_STEPS = 80
LIVE_R = 0.1          # amplitude threshold for a "live" unit at end of rollout
CHIRP_RATE = 0.005    # |mean dω per step| threshold to count as chirping


def rollout(model, data, steps=ROLLOUT_STEPS):
    """Manually evolve all units for `steps`, return (steps, D) arrays."""
    cell = model.cell
    with torch.no_grad():
        _, (r, phi, omega) = model(data)
        h = model.fc_in(model.conv(data))
        r_drive = cell.fc_r(h)
        omega_drive = cell.fc_omega(h)
        R, W = [], []
        for _ in range(steps):
            r = torch.clamp(r + cell.dt * (r * (1 - r**2) + r_drive), min=0.0)
            domega = (omega_drive + cell.alpha * (r - 1.0)
                      + cell.beta * torch.sin(phi) - cell.gamma * omega)
            omega = torch.clamp(omega + cell.dt * domega,
                                -cell.max_freq, cell.max_freq)
            phi = phi + cell.dt * omega
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
            R.append(r[0].cpu().numpy().copy())
            W.append(omega[0].cpu().numpy().copy())
    return np.array(R), np.array(W)   # (steps, D)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True,
                              download=True, transform=tfm),
                              batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False,
                             transform=tfm), batch_size=128, shuffle=False)

    model = ChirpMNIST(latent_dim=64, steps=8).to(device)

    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1], map_location=device))
        print(f"Loaded weights from {sys.argv[1]}")
    else:
        opt = optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(1, 6):
            train_epoch(model, device, train_loader, opt, ep)
            test(model, device, test_loader)

    model.eval()

    # Survey over several test images (chirp behaviour is input-dependent)
    n_imgs = 8
    imgs, _ = next(iter(test_loader))
    live_frac, chirp_frac, all_rates = [], [], []
    for i in range(n_imgs):
        R, W = rollout(model, imgs[i:i+1].to(device))
        live = R[-1] > LIVE_R                          # (D,)
        dw = np.diff(W, axis=0).mean(axis=0)           # mean dω/step per unit
        chirping = np.abs(dw) > CHIRP_RATE
        live_frac.append(live.mean())
        chirp_frac.append((live & chirping).sum() / max(live.sum(), 1))
        all_rates.append(dw)
        if i == 0:
            R0, W0, live0 = R, W, live

    all_rates = np.concatenate(all_rates)
    print(f"\n=== Population survey ({n_imgs} images, 64 units, "
          f"{ROLLOUT_STEPS} steps) ===")
    print(f"Live units (r_end > {LIVE_R}):        "
          f"{100*np.mean(live_frac):.1f}% ± {100*np.std(live_frac):.1f}%")
    print(f"Chirping among live (|dω/dt|>{CHIRP_RATE}): "
          f"{100*np.mean(chirp_frac):.1f}% ± {100*np.std(chirp_frac):.1f}%")
    print(f"Chirp-rate distribution: median |dω/step| = "
          f"{np.median(np.abs(all_rates)):.4f}, "
          f"max = {np.abs(all_rates).max():.4f}")

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs[0, 0].plot(R0)
    axs[0, 0].set_title(f"Amplitude, all 64 units "
                        f"(live: {int(live0.sum())}/64)")
    axs[0, 1].plot(W0)
    axs[0, 1].set_title("Frequency, all 64 units")
    axs[1, 0].hist(all_rates, bins=40)
    axs[1, 0].set_title("Mean dω/step per unit (all images)")
    axs[1, 0].axvline(0, color='k', lw=0.5)
    # spectrogram-style view of the live outputs
    out = R0 * np.cos(np.cumsum(W0 * 0.1, axis=0))
    axs[1, 1].imshow(out.T, aspect='auto', cmap='RdBu',
                     vmin=-1, vmax=1, interpolation='nearest')
    axs[1, 1].set_title("Output r·cos(φ), units × steps")
    axs[1, 1].set_xlabel("internal step"); axs[1, 1].set_ylabel("unit")
    plt.tight_layout()
    plt.savefig("figs/population_survey.png", dpi=110)
    print("Saved figs/population_survey.png")


if __name__ == '__main__':
    main()
