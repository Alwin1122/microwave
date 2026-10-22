"""Small convolutional net on the breast-window sinogram.

This is the transferable piece of Reimer and Pistorius, IEEE J-ERM 2022:
a convolution over time and antenna, not a dense net on a flattened window
(they found the dense net and logistic regression did not generalise).
The net is kept small because a fair UM-BMID split has a few dozen phantoms.
Antenna padding wraps around the circle. Training shifts that circle at
random; the shift does not change the tumour label.
"""

from __future__ import annotations

import numpy as np

# Adopt a replacement for the official first-scan model only when the fair
# phantom-held-out ROC mean clears the published nested result (0.706) by
# more than that result's run-to-run spread (0.019).
OFFICIAL_FIRST_SCAN_AUC = 0.706
ADOPT_IF_MEAN_AUC = 0.725


def circular_shift(batch: np.ndarray, offset: int) -> np.ndarray:
    """Roll the antenna axis. ``batch`` is (n, channel, time, antenna)."""
    if offset % batch.shape[-1] == 0:
        return batch
    return np.roll(batch, int(offset), axis=-1)


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


class _GapConv:
    """Built lazily so importing this module does not require PyTorch."""

    def __new__(cls):
        torch, nn = _torch()

        class GapConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 8, kernel_size=5, padding=(2, 0))
                self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=(1, 0))
                self.drop = nn.Dropout(0.4)
                self.fc = nn.Linear(16, 1)

            def forward(self, x):
                x = _wrap_antennas(x, 2)
                x = torch.relu(self.conv1(x))
                x = torch.max_pool2d(x, kernel_size=2)
                x = _wrap_antennas(x, 1)
                x = torch.relu(self.conv2(x))
                x = x.mean(dim=(2, 3))
                return self.fc(self.drop(x)).squeeze(1)

        return GapConv()


def _wrap_antennas(x, pad: int):
    return __import__("torch").cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    max_epochs: int = 40,
    patience: int = 6,
) -> np.ndarray:
    """Phantom-grouped early stop on the training phantoms, then score ``test_x``.

    Returns a tumour probability per test scan. The test phantoms are not
    used to choose the number of epochs.
    """
    torch, nn = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    y = np.asarray(train_y, dtype=np.int64)
    groups = np.asarray(train_groups)
    inner_train, inner_val = _inner_split(y, groups, seed)
    if inner_val.size < 8 or len(np.unique(y[inner_val])) < 2:
        epochs = 12
    else:
        epochs = _early_stop_epochs(
            train_x[inner_train], y[inner_train], train_x[inner_val], y[inner_val], seed, max_epochs, patience, torch, device
        )
    model = _fit_epochs(train_x, y, epochs, seed, torch, device, rng)
    return _probabilities(model, test_x, torch, device)


def _inner_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = len(set(groups))
    if n_groups < 3 or len(np.unique(y)) < 2:
        index = np.arange(len(y))
        return index, index[:0]
    splitter = StratifiedGroupKFold(n_splits=min(4, n_groups), shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    return train_idx, val_idx


def _early_stop_epochs(train_x, train_y, val_x, val_y, seed, max_epochs, patience, torch, device) -> int:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    model = _GapConv().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_auc, best_epoch, stale = -1.0, 1, 0
    for epoch in range(1, max_epochs + 1):
        _run_epoch(model, opt, train_x, train_y, torch, device, rng, train=True)
        with torch.no_grad():
            score = _probabilities(model, val_x, torch, device)
        auc = float(roc_auc_score(val_y, score))
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch


def _fit_epochs(train_x, train_y, epochs, seed, torch, device, rng):
    torch.manual_seed(seed)
    model = _GapConv().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        _run_epoch(model, opt, train_x, train_y, torch, device, rng, train=True)
    return model


def _run_epoch(model, opt, data, labels, torch, device, rng, train: bool) -> None:
    model.train(train)
    order = rng.permutation(len(labels))
    pos = max(int((labels == 1).sum()), 1)
    neg = max(int((labels == 0).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for start in range(0, len(order), 32):
        batch_i = order[start : start + 32]
        shift = int(rng.integers(0, data.shape[-1]))
        batch = circular_shift(np.asarray(data[batch_i]), shift)
        xb = torch.tensor(batch, dtype=torch.float32, device=device)
        yb = torch.tensor(labels[batch_i], dtype=torch.float32, device=device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()


def _probabilities(model, data, torch, device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(data), 64):
            xb = torch.tensor(np.asarray(data[start : start + 64]), dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0, dtype=float)
