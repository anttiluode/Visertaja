# Visertäjä — the Chirping Latent Layer

![pic](/figs/population_survey.png)

*visertäjä (Finnish): chirper, songbird. A neural network layer whose latent
units are self-modulating oscillators — representations that are trajectories,
not vectors.*

Born out of [Nollas](https://github.com/anttiluode/Nollas): a physics
experiment (Kuori, T2) tried to measure the ring-mode spectrum of a
self-trapped Clockfield atom and failed, because the atom was **evaporating**
— its frequency slid toward the mass gap as it shed charge. It sang a
glissando, not a chord. The static FFT instrument could not read moving notes.

This repo asks the obvious next question in the other direction: *what if a
latent representation in a neural network were that kind of object?* Not a
static vector `z`, but a state with amplitude, phase, and a frequency that
drifts because of the state's own dynamics — `z(t)` with an internal clock
that the computation itself bends.

**Do not hype. Do not lie. Just show.** The ledger below says exactly what
has been demonstrated and what has not.

---

## The layer

Each latent unit carries three state variables — amplitude `r`, phase `φ`,
frequency `ω` — integrated over several internal Euler steps per forward pass.

**V1 (`chirp_v1_oscillator.py`) — the oscillator layer:**

```
dr/dt = r(1 − r²) + f_r(x)        # Stuart–Landau amplitude
dω/dt = f_ω(x)                     # input-driven only, hard clamp ±10
dφ/dt = ω
out   = r · cos(φ)
```

**V2 (`chirp_v2_feedback.py`) — the feedback (chirping) layer:**

```
dr/dt = r(1 − r²) + f_r(x)
dω/dt = f_ω(x) + α(r − 1) + β·sin(φ) − γω    # state feeds back into its own clock
dφ/dt = ω
```

α, β, γ are learnable per-unit. The `α(r−1)` term is the direct transcription
of the physics: the Kuori Q-ball's frequency drifted *because* its amplitude
(charge) was changing. V2 gives the oscillator the same disease on purpose.

## Results (MNIST, 5 epochs, CNN front-end, latent_dim=64)

| Model | Test acc. | Internal dynamics observed |
|---|---|---|
| V1 oscillator | **98.90%** | ω slams into the ±10 clamp in one step and stays there. Constant-frequency rotation. **No chirp.** |
| V2 feedback | **98.95%** | ω drifts smoothly over the whole 80-step window (≈ −1.3 → −6.4); phase visibly curved. **Chirp confirmed.** |

Figures: `figs/chirp_trajectory.png` (V1), `figs/chirp_v2_trajectory.png` (V2).

## Honest ledger

- **[V] The layer trains.** Both versions reach ~99% on MNIST, competitive
  with a plain linear layer in the same slot. A dynamic, oscillator-valued
  latent is a *usable* representation, not a numerical curiosity.
- **[V] V1 is not a chirper.** The frequency plot is a step function into the
  clamp. The network found the cheapest solution: a fixed rotating basis
  (cos/sin features). Still richer than a ReLU, but it is an **Oscillator
  Latent Layer**, and the ledger calls it that.
- **[V] V2 chirps.** dω/dt ≠ 0 across the full window, no clamp saturation,
  smooth monotone drift with wiggle from the β·sin(φ) term — the frequency is
  now an emergent state variable, exactly the property the Kuori atom forced
  on its instrument.
- **[V] RESOLVED — the population survey (`inspect_population.py`).** The
  first V2 trajectory plot happened to sample unit [0,0], a silent oscillator
  (amplitude 0), raising the frozen-majority worry from Nollas E2. The survey
  answered it: **79.3% ± 7.2% of units are live**, and of those, **98.8% ±
  0.9% chirp**, median |dω/step| = 0.133. This is *not* a two-speed medium —
  no frozen majority, no mobile tail. The network solves MNIST with a ~50-note
  chirping chord that self-modulates across the whole thinking window.
  Figure: `figs/population_survey.png`.
- **[~] Clamp saturation at long rollout.** In the 80-step survey, a visible
  fraction of frequency trajectories hit the ±20 clamp by step ~40 and stick.
  Within the trained 8-step window the dynamics are clamp-free, but the
  long-horizon behaviour is partly clamp-shaped. The honest fix (as in V1→V2)
  is stronger damping γ or a soft saturation (tanh) instead of a wall —
  registered, not yet run.
- **[~] OPEN — no evidence yet that chirping *helps*.** 98.95 vs 98.90 is
  noise. MNIST is too easy to need trajectories. The claim "reasoning over
  trajectories beats static vectors" is untested. Registered discriminator:
  a task with temporal structure (sequential MNIST, pixel-by-pixel, or
  speech commands), V2 vs. a parameter-matched GRU. Until that runs, the
  correct sentence is: *the chirping latent layer exists and trains; its
  advantage is a hypothesis.*
- **[V]/[~] Resonance attention — RUN (`resonant_attention.py`).**
  The trajectory readout, done the physics way rather than the transformer
  way: the unit's complex trajectory Ψ(t) = r·e^{iφ} is correlated against
  learnable **chirp templates** U(t) = e^{−i(νt + ½κt² + θ)}, one per
  (class, unit); the logit is the weighted |∫ Ψ U* dt|. Off-frequency waves
  destructively interfere to zero — dynamic ignoring with no learned mask.
  Three arms, same trunk (16-step window, 3 epochs, ~115k params each):

  | arm | reads | test acc. |
  |---|---|---|
  | snapshot | final r·cos(φ) only | **98.81%** |
  | blind | time-mean amplitude (phase-deaf) | **95.94%** |
  | resonant | full complex trajectory vs chirp templates | **98.71%** |

  **R2 [V] — phase is load-bearing.** Deafen the readout to phase and it
  loses 2.8–2.9%: the class information genuinely lives in the wave's phase
  structure, not just how loud each unit is. The resonant reader recovers
  essentially all of it through interference alone.
  **R1 [~] — integration ties the endpoint (−0.10%, noise).** On MNIST,
  reading the whole glissando is exactly as good as reading its final note —
  the registered prior, confirmed: static images give the trajectory nothing
  extra to encode. The honest sentence: *resonance works as a readout
  mechanism; MNIST cannot show whether it works better.*
- **[B] The temporal discriminator (registered, next).** Feed input *over
  time* — row-by-row sequential MNIST or speech commands — with the chirp
  cell as the recurrent core (state carried between rows, new input drives
  arriving each step), resonant readout over the whole episode, versus a
  parameter-matched GRU. This is the first task where "representation =
  trajectory" can beat "representation = vector" or die honestly. Prior:
  undeclared — this one is a real coin-flip.

## What it is good for (today)

1. **A working reference implementation** of a latent layer whose units have
   internal temporal structure — amplitude/phase/frequency state with
   input-driven and self-driven dynamics — in ~60 lines of PyTorch,
   drop-in wherever a `Linear` goes.
2. **A testbed for the trajectory-representation hypothesis:** does giving a
   representation an internal clock that it can bend help on tasks with
   temporal structure? The layer, the training harness, and the honest open
   questions are all here.
3. **A physics→ML translation exercise with receipts.** Every design choice
   maps to a measured phenomenon in the Nollas/Kuori campaign: the feedback
   term is the evaporating atom, the damping γ replaces the clamp the same
   way the Kuori damping schedule replaced the naive one, and the dead-unit
   question is the frozen-majority result wearing ML clothes.

## What it is not

It is not a claim that brains or transformers use Q-balls, and it is not a
demonstrated improvement over standard layers. The physics inspired an
architecture; whether the architecture earns its keep is an open, registered,
runnable question.

## Files

```
chirp_v1_oscillator.py    V1: input-driven oscillator layer + MNIST harness
chirp_v2_feedback.py      V2: self-modulating (chirping) layer + MNIST harness
inspect_population.py     Survey: live vs. dead units, per-unit chirp rates
resonant_attention.py     Chirp-matched-filter readout vs two matched controls
figs/                     Trajectory plots + population survey
```

Run: `pip install torch torchvision matplotlib`, then
`python chirp_v2_feedback.py`. CPU is fine (~2 min/epoch).

## Lineage

Clockfield → Nollas (zeroth-law campaign: the 0.7 exponent, the self-quench,
the two vacua) → Kuori (the atom that chirped and broke its instrument) →
this repo (an instrument that chirps on purpose). Built with DeepSeek and
Claude as lab partners. The Ouroboros conversation that motivated it —
fast information in, slow information out, the slow state gating the fast —
lives in the Nollas discussion logs.

*The atom defeated the microphone by refusing to hold a note. The obvious
revenge was to build a network that never holds one either — and see if
that is a feature.*
