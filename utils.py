from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 



#---------------------------------data preprocessing-----------------------------------

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
    


#--------------------------------- plotting -----------------------------------   

def plot_point_cloud(
        pc,
        title: str | None = None,
        elev: float = 30,
        azim: float = 45,
        s: float = 2,        
        c: str | np.ndarray = "tab:blue",
):
        """
        Visualise a single point cloud.

        Parameters
        ----------
        pc : (N, 3) np.ndarray | torch.Tensor
            xyz coordinates.
        title : str | None
            Optional plot title.
        elev, azim : float
            View angles (matplotlib convention).
        figsize : tuple[int, int]
            Size of the figure in inches.
        s : float
            Scatter marker size.
        c : str | array-like
            Marker colour(s). Can be a single colour or per-point array.
        """
        # Convert torch → numpy if needed
        if "torch" in str(type(pc)):
            pc = pc.detach().cpu().numpy()

        assert pc.ndim == 2 and pc.shape[1] == 3, "input must be (N, 3)"

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=s, c=c, depthshade=False)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if title is not None:
            ax.set_title(title)

        ax.view_init(elev=elev, azim=azim) 
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(f'{title}.png')


def visualize_classes(classes_to_viz: list):
    for class_to_viz in classes_to_viz:
        txt_path = f"modelnet40_normal_resampled/{class_to_viz}/{class_to_viz}_0001.txt"
        pointcloud = _load_point_cloud(txt_path=txt_path, num_points=1024)
        plot_point_cloud(pointcloud, title=class_to_viz, elev=20, azim=-60)


visualize_classes(['bottle', 'chair', 'airplane', 'monitor'])


# ------------------------------------------------------------------
train_ds = ModelNet40(
    root_dir="modelnet40_normal_resampled",
    split_txt="modelnet40_train.txt",
    num_points=256
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



