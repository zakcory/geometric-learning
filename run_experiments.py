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
import matplotlib.pyplot as plt
import pandas as pd

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
    patience=20,
    lr=1e-3,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=128
):
    """
    Train model until early-stop or exceeding 0.5 hours run-time.
    Returns (status_str, test_acc, loss_steps, run_time).
    """

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    best_val, best_state = -float("inf"), None
    start = time.time()
    wait = 0
    loss_steps = list()
    bar = tqdm(range(1000000))

    for epoch in bar:      
        # --- time check ----------------------------------------------------
        
        if time.time() - start > max_runtime_s:
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
            wait = 0

        else:
            wait += 1
            if wait >= patience:
                break  # early-stop according to the val set

        bar.set_postfix({'Loss':loss.item()})

        if ((epoch+1) % 10) == 0:
                print(f"===============Epoch: {epoch+1}, Loss: {loss.item()}, Val Accuracy: {best_val}=================")

    # restore best & test ----------------------------------------------------
    if best_state is not None:                
        model.load_state_dict(best_state)
    test_acc = accuracy(model, test_loader, device)
    return f"Total training Epochs={epoch+1}", test_acc, loss_steps, (time.time() - start)


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
        "Linear Equiv. Layers": lambda: EquivariantLinearMLP(),
        "Data Augm.": lambda: DataAugmentationMLP(),
    }

    results = {}
    losses = {}

    for name, build in models.items():
        print(f"\n===== {name} ====")
        results[name] = {}
        losses[name] = {}
        for k in sizes:
            print(f"  ► {k} samples")

            # build datasets ------------------------------------------------
            subset = make_subset(train_full, per_cls, k, seed=42)
            val_len = max(1, int(0.15 * len(subset)))
            train_len = len(subset) - val_len
            train_sub, val_sub = random_split(subset, [train_len, val_len], generator=torch.Generator().manual_seed(0))

            # model ---------------------------------------------------------
            model = build()

            status, test_acc, loss_steps, run_time = train_one_run(
                model, train_sub, val_sub, test_loader
            )
            print(f"{status} | Test Acc = {test_acc:.3f}")
            results[name][k] = {"acc": test_acc, "status": status, "time": run_time}
            losses[name][k] = loss_steps

    # summary ---------------------------------------------------------------
    print("\n===== SUMMARY =====")
    for name, table in results.items():
        print(name)
        for k in sizes:
            status = table[k]["status"]
            acc = table[k]["acc"]
            print(f"  k={k}: {status:<8}  Test Acc={acc:.3f}")
    print("===================")

    # ---------- accuracy curve ---------------------------------------
    sizes = [5, 10, 50]
    plt.figure(figsize=(7,4))
    for arch, tbl in results.items():
        accs = [tbl[k]["acc"] for k in sizes]
        plt.plot(sizes, accs, marker='o', label=arch)
    plt.xlabel("training examples per class")
    plt.ylabel("test accuracy")
    plt.title("Accuracy vs training size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")

    # ---------- runtime table ----------------------------------------
    rows=[]
    for arch, tbl in results.items():
        for k in sizes:
            rows.append({"Architecture": arch,
                        "Samples per Class":  k,
                        "Runtime (s)":  f"{tbl[k]['time']:.1f}",
                        "Acc":          f"{tbl[k]['acc']:.3f}"})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Architecture", columns="Samples per Class",
                    values=["Runtime (s)", "Acc"])
    print("\nRuntime & Accuracy table\n")
    print(pivot.to_string())
    pivot.to_csv("runtime_table.csv")


if __name__ == "__main__":
    run_all()
