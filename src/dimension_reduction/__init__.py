from typing import Optional, List, Dict, Any
from torch.nn import Module
from src.utils import read_image

from tqdm import tqdm


class DimensionReduction:
    def __init__(
        self,
        features: List,
        images: List,
        labels: List,
        label_names: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """_summary_

        Args:
            feature_extractor (Module): cnn model for getting feature of images
            images (List): image path list
            labels (List): parse `images` to label
            label_names (Optional[Dict[Any, Any]], optional): _description_. Defaults to None.
            transform (Optional[Any], optional): _description_. Defaults to None.
        """

        self.features = features
        self.images = images
        self.labels = labels
        self.label_names = (
            label_names if label_names else {i: str(i) for i in range(len(set(labels)))}
        )

        self.model = None  # tSNE or PCA model

    def train_model(self):
        pass

    def save_plot(self):
        pass
