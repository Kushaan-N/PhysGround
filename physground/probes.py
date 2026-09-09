"""Linear probes: ridge, logistic, the spatial contact probe, and controls (spec 8).

Linear only for the headline results (spec 3.1). An MLP probe invites the
objection that the probe learned the task rather than read it out, and the whole
claim being made is about what is *linearly available* in the frozen
representation.

Why ridge is solved by eigendecomposition
-----------------------------------------
The main grid asks for 6 regression targets x 17 alphas x 5 inner folds for
every (encoder, layer, view, seed) cell. Fitting those separately is 510
independent least-squares solves per cell. But the design matrix is the same for
all of them, and a ridge solution for *any* alpha and *any* target follows from
one eigendecomposition of the Gram matrix:

    G = Xc'Xc = V diag(w) V'   =>   beta(alpha) = V (V'Xc'Yc) / (w + alpha)

So one eigh of a 768x768 matrix (~50 ms) yields every alpha and every target at
once, and validation predictions cost one matrix product each. Per-fold Gram
matrices come from subtracting the fold's own contribution from the full
training Gram, so five folds cost roughly two full Gram computations rather than
five. The result is numerically identical to fitting each separately -- only the
arithmetic path differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .splits import grouped_folds

__all__ = [
    "ProbePrediction",
    "RIDGE_ALPHAS",
    "LOGISTIC_CS",
    "MultiRidgeCV",
    "LogisticProbe",
    "shuffle_labels_across_scenes",
    "spatial_contact_probe",
]

#: spec 8.1
RIDGE_ALPHAS = np.logspace(-3, 5, 17)
#: spec 8.2
LOGISTIC_CS = np.logspace(-4, 4, 9)

_EPS = 1e-12


@dataclass
class ProbePrediction:
    """Raw held-out predictions, persisted verbatim (spec 0.4).

    Every summary statistic in the paper is recomputed from these arrays by a
    separate script, so no number in a table depends on re-running a model.
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    scene_index: np.ndarray
    hyperparameter: float
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #

def _standardiser(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and scale from the training rows only (spec 8.1).

    Constant columns get scale 1 rather than 0. A random-init encoder can easily
    produce a dead unit, and dividing by its zero standard deviation would put
    NaNs through the entire probe -- turning a harmless degenerate feature into a
    missing row in the results table.
    """
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < _EPS] = 1.0
    return mean, scale


# --------------------------------------------------------------------------- #
# Ridge
# --------------------------------------------------------------------------- #

class MultiRidgeCV:
    """Ridge with per-target alpha chosen by grouped CV, all targets at once.

    Equivalent to running ``RidgeCV`` per target with a scene-grouped inner CV,
    computed through a shared eigendecomposition (see the module docstring).
    ``tests/test_probes.py`` asserts agreement with scikit-learn's ``Ridge``.
    """

    def __init__(self, alphas: Sequence[float] = RIDGE_ALPHAS, n_folds: int = 5, seed: int = 0):
        self.alphas = np.asarray(alphas, dtype=float)
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.best_alpha_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.cv_scores_: np.ndarray | None = None

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _moments(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Raw (uncentred) second moments, so folds can be formed by subtraction."""
        return x.T @ x, x.T @ y, x.sum(axis=0), y.sum(axis=0), x.shape[0]

    @staticmethod
    def _solve(second_moment: np.ndarray, cross: np.ndarray, sum_x: np.ndarray,
               sum_y: np.ndarray, n: int, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ridge coefficients and intercepts for every alpha, from raw moments.

        Centring is applied here rather than to the data, which is what allows a
        fold's moments to be obtained by subtraction. The intercept is never
        penalised -- ridge that shrinks the intercept would bias predictions
        toward zero rather than toward the mean, and R2 is measured against the
        mean.
        """
        mean_x = sum_x / n
        mean_y = sum_y / n
        gram = second_moment - n * np.outer(mean_x, mean_x)
        centred_cross = cross - n * np.outer(mean_x, mean_y)

        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        # eigh on a PSD matrix can return small negative values from rounding.
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        projected = eigenvectors.T @ centred_cross                       # (d, k)

        coefs = np.empty((alphas.size, gram.shape[0], centred_cross.shape[1]))
        intercepts = np.empty((alphas.size, centred_cross.shape[1]))
        for i, alpha in enumerate(alphas):
            beta = eigenvectors @ (projected / (eigenvalues + alpha)[:, None])
            coefs[i] = beta
            intercepts[i] = mean_y - mean_x @ beta
        return coefs, intercepts

    # -- api ---------------------------------------------------------------- #

    def fit(self, x: np.ndarray, y: np.ndarray, scene_index: np.ndarray) -> "MultiRidgeCV":
        x = np.asarray(x, dtype=np.float64)
        y = np.atleast_2d(np.asarray(y, dtype=np.float64).T).T if y.ndim == 1 else np.asarray(y, np.float64)

        self._mean, self._scale = _standardiser(x)
        xs = (x - self._mean) / self._scale

        full = self._moments(xs, y)
        scores = np.zeros((self.alphas.size, y.shape[1]))
        n_folds = 0

        for train_rows, val_rows in grouped_folds(scene_index, self.n_folds, self.seed):
            xv, yv = xs[val_rows], y[val_rows]
            hold = self._moments(xv, yv)
            fold = tuple(f - h for f, h in zip(full, hold))
            coefs, intercepts = self._solve(*fold, self.alphas)

            # Score every alpha with one product against the fold's projections.
            for i in range(self.alphas.size):
                pred = xv @ coefs[i] + intercepts[i]
                residual = ((yv - pred) ** 2).sum(axis=0)
                total = ((yv - yv.mean(axis=0)) ** 2).sum(axis=0)
                scores[i] += np.where(total > _EPS, 1.0 - residual / np.maximum(total, _EPS), 0.0)
            n_folds += 1

        self.cv_scores_ = scores / max(n_folds, 1)
        best = np.argmax(self.cv_scores_, axis=0)
        self.best_alpha_ = self.alphas[best]

        coefs, intercepts = self._solve(*full, self.alphas)
        self.coef_ = np.stack([coefs[best[k], :, k] for k in range(y.shape[1])], axis=1)
        self.intercept_ = np.array([intercepts[best[k], k] for k in range(y.shape[1])])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=np.float64) - self._mean) / self._scale
        return xs @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------- #
# Logistic
# --------------------------------------------------------------------------- #

class LogisticProbe:
    """L2 logistic regression with C chosen by grouped CV on balanced accuracy.

    The C path is walked in ascending order with warm starts: each fit begins
    from the previous solution, which is close, so the whole path costs little
    more than a few cold fits. Balanced accuracy is the selection criterion
    rather than accuracy, matching what is reported (spec 8.2) and preventing the
    search from selecting a model that predicts the majority class everywhere.
    """

    def __init__(self, cs: Sequence[float] = LOGISTIC_CS, n_folds: int = 5, seed: int = 0,
                 max_iter: int = 500):
        self.cs = np.asarray(cs, dtype=float)
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.best_c_: float | None = None
        self.model_ = None

    def _make(self):
        from sklearn.linear_model import LogisticRegression
        # `penalty` is left at its default rather than passed as "l2". The
        # default has always been L2, and scikit-learn 1.8 deprecated passing it
        # explicitly, so naming it earns a FutureWarning per fit now and a
        # TypeError in 1.10 -- with no change in the model being fit.
        return LogisticRegression(solver="lbfgs", max_iter=self.max_iter,
                                  warm_start=True, tol=1e-4)

    def fit(self, x: np.ndarray, y: np.ndarray, scene_index: np.ndarray) -> "LogisticProbe":
        from .stats import balanced_accuracy

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self._mean, self._scale = _standardiser(x)
        xs = (x - self._mean) / self._scale

        scores = np.zeros(self.cs.size)
        n_folds = 0
        for train_rows, val_rows in grouped_folds(scene_index, self.n_folds, self.seed):
            if np.unique(y[train_rows]).size < 2:
                continue
            model = self._make()
            for i, c in enumerate(self.cs):
                model.C = float(c)
                model.fit(xs[train_rows], y[train_rows])
                scores[i] += balanced_accuracy(y[val_rows], model.predict(xs[val_rows]))
            n_folds += 1

        self.cv_scores_ = scores / max(n_folds, 1)
        self.best_c_ = float(self.cs[int(np.argmax(self.cv_scores_))]) if n_folds else 1.0

        final = self._make()
        final.C = self.best_c_
        final.fit(xs, y)
        self.model_ = final
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=np.float64) - self._mean) / self._scale
        return self.model_.predict(xs)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=np.float64) - self._mean) / self._scale
        return self.model_.decision_function(xs)


# --------------------------------------------------------------------------- #
# Selectivity control (spec 8.5)
# --------------------------------------------------------------------------- #

def shuffle_labels_across_scenes(labels: np.ndarray, scene_index: np.ndarray,
                                 seed: int = 0) -> np.ndarray:
    """Permute labels between scenes *and* within each scene's block.

    Both permutations are necessary, and each fixes a case the other misses.

    *Across scenes* handles per-scene targets. ``mass`` and ``friction_slide``
    are constant within a scene, so shuffling only within a scene would leave
    every label exactly where it was.

    *Within a scene* handles per-frame targets. Frames are captured at fixed
    phases (spec 5.2), so ``contact_state`` is literally
    ``[0,0,0,1,1,1,1,0,0,0]`` in every scene and ``ee_obj_dist`` follows the
    same stereotyped arc. Swapping whole blocks between scenes therefore
    permutes identical vectors and changes nothing: measured on the pilot, a
    block-only control scored 0.869 balanced accuracy on ``contact_state`` and
    R2 0.688 on ``ee_obj_dist`` -- it was silently re-running the real task and
    reporting it as a control.

    Together they preserve the label multiset exactly while breaking the link
    between an image and its label. What they deliberately preserve is
    scene-level *memorisability*: each scene keeps a fixed label vector, so if a
    scene appeared on both sides of the split a probe could still learn it and
    the control would rise above chance. That is precisely the leak this control
    exists to detect (spec 7.4, 8.5).
    """
    labels = np.asarray(labels)
    scene_index = np.asarray(scene_index)
    order = np.lexsort((np.arange(labels.size), scene_index))
    scenes = scene_index[order]
    boundaries = np.flatnonzero(np.r_[True, scenes[1:] != scenes[:-1]])
    blocks = np.split(order, boundaries[1:])

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(blocks))

    shuffled = labels.copy()
    for destination, source in enumerate(permutation):
        donor = labels[blocks[source]]
        target_rows = blocks[destination]
        # Resample positions so ragged blocks still fill, then shuffle within.
        if donor.size != target_rows.size:
            donor = donor[np.arange(target_rows.size) % donor.size]
        shuffled[target_rows] = rng.permutation(donor)
    return shuffled


# --------------------------------------------------------------------------- #
# Spatial contact probe (spec 8.3)
# --------------------------------------------------------------------------- #

def spatial_contact_probe(patches_train: np.ndarray, y_train: np.ndarray,
                          patches_test: np.ndarray, *, weight_decay: float = 1e-3,
                          epochs: int = 30, batch_size: int = 256, lr: float = 3e-3,
                          seed: int = 0, device: str | None = None) -> np.ndarray:
    """Per-patch linear probe, max-pooled over patches. Returns test scores.

    Tests whether contact is encoded *somewhere* spatially even when the pooled
    views wash it out -- a real possibility, and the first thing a reviewer will
    ask about a null result on the pooled features (spec 8.3).

    The readout stays linear in the patch features; only the pooling is
    nonlinear, which is what makes it a localisation test rather than a more
    powerful probe. Max-pooling makes this multiple-instance learning: the
    gradient reaches only the arg-max patch per frame, so training is
    deliberately slow and heavily regularised rather than aggressive.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    x_train = torch.as_tensor(np.asarray(patches_train, dtype=np.float32))
    labels = torch.as_tensor(np.asarray(y_train, dtype=np.float32))

    # Standardise per feature over all patches of the training frames.
    mean = x_train.mean(dim=(0, 1), keepdim=True)
    std = x_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

    weight = torch.zeros(x_train.shape[-1], 1, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([weight, bias], lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n = x_train.shape[0]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            rows = order[start:start + batch_size]
            batch = ((x_train[rows] - mean) / std).to(device)
            logits = (batch @ weight).squeeze(-1) + bias        # (b, patches)
            pooled = logits.max(dim=1).values
            loss = loss_fn(pooled, labels[rows].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    with torch.no_grad():
        x_test = torch.as_tensor(np.asarray(patches_test, dtype=np.float32))
        scores = []
        for start in range(0, x_test.shape[0], batch_size):
            batch = ((x_test[start:start + batch_size] - mean) / std).to(device)
            logits = (batch @ weight).squeeze(-1) + bias
            scores.append(logits.max(dim=1).values.cpu().numpy())
    return np.concatenate(scores)
