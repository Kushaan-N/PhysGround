"""A small DINO-WM-style latent predictor, for H5 (spec 2, Exp G).

H5 asks whether probe decodability *predicts downstream latent-dynamics error*:
if contact state is not linearly available in a representation, a dynamics model
built on that representation should be worse at predicting what happens next.
Spec 2 marks it a stretch goal with predicted outcome "unknown", to run only if
Experiments 0-F complete cleanly.

What this is
------------
The smallest thing that is still recognisably the design under test: a two-layer
MLP trained to predict the *next latent* from the current latent and an action,

    z_{t+1} ~= f(z_t, a_t)

with the encoder frozen and no pixel reconstruction anywhere. It is not a world
model — there is no planning, no multi-step rollout, no goal scoring. The claim
H5 needs is comparative, across representations of the same scenes, so what
matters is that every cell gets the same predictor and the same data.

The action
----------
``a_t`` is the realised end-effector displacement between consecutive captured
frames, read from the simulator rather than from the commanded trajectory. The
position actuators have real tracking error under load, so the commanded
displacement is not what happened.

Frames are captured at phase fractions, not fixed timesteps (spec 5.2), so the
interval between consecutive frames varies. ``dt`` is therefore appended to the
action: without it the predictor is asked to infer how much time passed from the
latent alone, and would be penalised for a property of the capture schedule
rather than of the representation.

Normalisation
-------------
Prediction error is reported as R2 against predicting ``z_{t+1} = z_t`` — the
stationary baseline — rather than as raw MSE. Latent scales differ by encoder
and by layer, so raw MSE is not comparable across exactly the cells H5 wants to
compare, and the correlation it reports would partly be a correlation with
feature norm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LatentTransition", "build_transitions", "fit_latent_predictor"]


@dataclass
class LatentTransition:
    """Consecutive-frame transitions within scenes."""
    z_now: np.ndarray        # (N, D)
    z_next: np.ndarray       # (N, D)
    action: np.ndarray       # (N, 4) end-effector displacement plus dt
    scene_index: np.ndarray  # (N,)


def build_transitions(x: np.ndarray, targets: dict[str, np.ndarray],
                      scene_index: np.ndarray) -> LatentTransition:
    """Pair each frame with its successor, never crossing a scene boundary.

    Crossing a boundary would ask the predictor to model a discontinuous jump
    between unrelated scenes, which is not a dynamics failure and would add the
    same constant penalty to every cell.
    """
    scene_index = np.asarray(scene_index)
    same_scene = scene_index[:-1] == scene_index[1:]
    rows = np.flatnonzero(same_scene)

    ee = np.column_stack([targets["ee_pos_x"], targets["ee_pos_y"]]).astype(np.float64)
    time = np.asarray(targets["time"], dtype=np.float64)

    displacement = ee[rows + 1] - ee[rows]
    dt = (time[rows + 1] - time[rows])[:, None]
    action = np.hstack([displacement, dt, np.ones_like(dt)])

    return LatentTransition(
        z_now=np.asarray(x, dtype=np.float64)[rows],
        z_next=np.asarray(x, dtype=np.float64)[rows + 1],
        action=action,
        scene_index=scene_index[rows],
    )


def fit_latent_predictor(transitions: LatentTransition, train_mask: np.ndarray,
                         test_mask: np.ndarray, *, hidden: int = 512, epochs: int = 60,
                         batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4,
                         seed: int = 0, device: str | None = None) -> dict:
    """Train the predictor on train scenes and score it on held-out scenes.

    Returns R2 against the stationary baseline, plus that baseline's own MSE so
    a reader can see how much motion there was to predict at all.
    """
    import torch

    from .encoders.base import pick_device

    device = pick_device(device)
    torch.manual_seed(seed)

    # Standardise latents on training rows only; the same statistics are applied
    # to inputs and targets so the predictor works in one consistent space.
    z_train = transitions.z_now[train_mask]
    mean = z_train.mean(axis=0, keepdims=True)
    scale = z_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0

    def prepare(values):
        return torch.as_tensor((values - mean) / scale, dtype=torch.float32)

    action_scale = np.abs(transitions.action[train_mask]).max(axis=0, keepdims=True)
    action_scale[action_scale < 1e-8] = 1.0

    inputs_train = torch.cat([prepare(transitions.z_now[train_mask]),
                              torch.as_tensor(transitions.action[train_mask] / action_scale,
                                              dtype=torch.float32)], dim=1)
    targets_train = prepare(transitions.z_next[train_mask])
    inputs_test = torch.cat([prepare(transitions.z_now[test_mask]),
                             torch.as_tensor(transitions.action[test_mask] / action_scale,
                                             dtype=torch.float32)], dim=1)
    targets_test = prepare(transitions.z_next[test_mask])

    dimension = targets_train.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(inputs_train.shape[1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, dimension),
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)

    n = inputs_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            rows = order[start:start + batch_size]
            prediction = model(inputs_train[rows].to(device))
            loss = torch.nn.functional.mse_loss(prediction, targets_train[rows].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        schedule.step()

    model.eval()
    with torch.no_grad():
        predicted = model(inputs_test.to(device)).cpu().numpy()

    truth = targets_test.numpy()
    stationary = torch.cat([prepare(transitions.z_now[test_mask])]).numpy()

    model_mse = float(np.mean((truth - predicted) ** 2))
    stationary_mse = float(np.mean((truth - stationary) ** 2))
    return {
        "r2_vs_stationary": float(1.0 - model_mse / stationary_mse) if stationary_mse > 0 else float("nan"),
        "model_mse": model_mse,
        "stationary_mse": stationary_mse,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "latent_dim": int(dimension),
    }
