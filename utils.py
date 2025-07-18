from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_split_list(list_path: Path) -> list[str]:
    """Return the file-stems (e.g. 'airplane_001') from a split list file (to seperate train, test)"""
    with list_path.open() as f:
        return [line.strip() for line in f if line.strip()]


def _load_point_cloud(txt_path: Path, num_points: int | None = None) -> np.ndarray:
    """
    Load a ModelNet .txt file and return (N,3) xyz coordinates.

    Parameters
    ----------
    txt_path : Path
        Full path to the .txt file.
    num_points : int | None
        If given, keep only the first `num_points` rows.
    """
    data = np.loadtxt(txt_path, delimiter=',')         
    if num_points is not None:
        data = data[:num_points]
    return data[:, :3].astype(np.float32)               # keep xyz only


class ModelNet40(Dataset):
    """
    ModelNet40 loader that follows the train/test split files.
    """

    def __init__(
        self,
        root_dir,
        split_txt,
        num_points=256,
        transform=None      
    ):
        
        self.root = Path(root_dir)
        self.names = _read_split_list(self.root.joinpath(Path(split_txt)))
        self.num_points = num_points
        self.transform = transform

        classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.cls2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx):
        stem = self.names[idx]
        cls = stem.split('_', 1)[0]
        file = self.root / cls / f"{stem}.txt"

        pc = _load_point_cloud(file, self.num_points)     # (n,3)
        if self.transform is not None:
            pc = self.transform(pc)

        label = self.cls2idx[cls]
        return torch.from_numpy(pc), label


# ------------------------------------------------------------------
train_ds = ModelNet40(
    root_dir="modelnet40_normal_resampled",
    split_txt="modelnet40_train.txt"
)
test_ds  = ModelNet40(
    root_dir="modelnet40_normal_resampled",
    split_txt="modelnet40_test.txt"
)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=32, shuffle=True, drop_last=True
)

print("Train Obj Shape: ", train_ds[0][0].shape)
print("Train Len: ", len(train_ds))

print("Test Obj Shape: ", test_ds[0][0].shape)
print("Test Len: ", len(test_ds))

