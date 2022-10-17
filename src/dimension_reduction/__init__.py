from .pca import PCA
from .tsne import tSNE

__all__ = [
    "PCA",
    "tSNE",
]

from typing import Optional, List, Dict, Any
from torch.nn import Module


class DimensionReduction:
    def __init__(
        self,
        feature_extractor: Module,
        images: List,
        labels: List,
        label_names: Optional[Dict[Any, Any]] = None,
        transform: Optional[Any] = None,
    ) -> None:

        self.feature_extractor = feature_extractor
        self.transform = transform

        self.images = images
        self.labels = labels
        self.label_names = (
            label_names if label_names else {i: str(i) for i in range(len(set(labels)))}
        )

        self.transform = transform if transform else None

        self.features = None

    def feature_extraction(self):
        pass

    def train_model(self):
        pass

    def save_graph(self):
        pass
