from models import *
from utils import *

import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
from tqdm import tqdm

# ----------------------------- helpers -------------------------------------

def per_class_indices(dataset):
    """Return a dict {class_idx: [dataset_indices]}."""
    buckets = {}
    for i, (_, label) in enumerate(dataset):
        buckets.setdefault(int(label), []).append(i)
    return buckets


def make_subset(dataset, per_class, samples_per_class, seed=0):
    """
    Return a Subset containing `samples_per_class` examples for every class.
    """
    rng = random.Random(seed)
    chosen = []
    for cls, idxs in per_class.items():
        rng.shuffle(idxs)
        chosen.extend(idxs[:samples_per_class])
    return Subset(dataset, chosen)


def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ----------------------------- training loop -------------------------------

def train_one_run(
    model,
    train_subset,
    val_subset,
    test_loader,
    *,
    max_runtime_s=1800,          # 0.5 hour
    patience=10,
    lr=1e-3,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Train `model` until early-stop or `max_runtime_s`.
    Returns (status_str, test_acc).
    """

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4)

    best_val, best_state = 0.0, None
    start = time.time()
    loss_steps = list()
    bar = tqdm(range(100))

    for epoch in bar:          # effectively "while True"
        # --- time check ----------------------------------------------------
        
        if time.time() - start > max_runtime_s:
            print("HERE")
            status = "OOT"                  # out of time
            if best_state is not None:
                model.load_state_dict(best_state)
            test_acc = accuracy(model, test_loader, device)
            return status, test_acc

        # ------------------- one epoch ------------------------------------
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss_steps.append(loss.item())
            loss.backward()
            opt.step()
            

        # ------------------- validation -----------------------------------
        val_acc = accuracy(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        bar.set_postfix({'Loss':loss.item()})

        if ((epoch+1) % 10) == 0:
                print(f"===============Epoch: {epoch+1}, Loss: {loss.item()}, Val Accuracy: {best_val}=================")

    # restore best & test ----------------------------------------------------
    model.load_state_dict(best_state)
    test_acc = accuracy(model, test_loader, device)
    return f"epochs={epoch+1}", test_acc, loss_steps


# ----------------------------- experiment grid -----------------------------

def run_all():
    # ---------- dataset ----------------------------------------------------
    root = "modelnet40_normal_resampled"
    train_full = ModelNet40(root, "modelnet40_train.txt", num_points=256)
    test_set = ModelNet40(root, "modelnet40_test.txt",  num_points=256)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    per_cls = per_class_indices(train_full)

    sizes = [5, 10, 50]          # samples per class
    models = {
        "Canonization":         lambda: CanonizationMLP(),
        "Full-Symmetrization":  lambda: SymmetrizationMLP(),
        "Sampled-Symmetrization": lambda: SampledSymmetrizationMLP(),
    }

    results = {}

    for name, build in models.items():
        print(f"\n===== {name} ====")
        results[name] = {}
        for k in sizes:
            print(f"  ► {k} samples")

            # build datasets ------------------------------------------------
            subset = make_subset(train_full, per_cls, k, seed=42)
            val_len = max(1, int(0.15 * len(subset)))
            train_len = len(subset) - val_len
            train_sub, val_sub = random_split(subset, [train_len, val_len], generator=torch.Generator().manual_seed(0))

            # model ---------------------------------------------------------
            model = build()

            status, test_acc, loss_steps = train_one_run(
                model, train_sub, val_sub, test_loader,
                max_runtime_s=1800, patience=10
            )
            print(f"{status:>8} | test acc = {test_acc:.3f}")
            results[name][k] = (status, test_acc)

    # summary ---------------------------------------------------------------
    print("\n===== SUMMARY =====")
    for name, table in results.items():
        print(name)
        for k in sizes:
            status, acc = table[k]
            print(f"  k={k:>2}: {status:<8}  acc={acc:.3f}")
    print("===================")


if __name__ == "__main__":
    run_all()
