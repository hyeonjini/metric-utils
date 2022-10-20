from typing import Optional, List, Dict, Any
from torch.nn import Module
from src.utils import read_image

from tqdm import tqdm


class DimensionReduction:
    def __init__(
        self,
        feature_extractor: Module,
        images: List,
        labels: List,
        label_names: Optional[Dict[Any, Any]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        """_summary_

        Args:
            feature_extractor (Module): cnn model for getting feature of images
            images (List): image path list
            labels (List): parse `images` to label
            label_names (Optional[Dict[Any, Any]], optional): _description_. Defaults to None.
            transform (Optional[Any], optional): _description_. Defaults to None.
        """

        self.feature_extractor = feature_extractor
        self.feature_extractor
        self.transform = transform

        self.images = images
        self.labels = labels
        self.label_names = (
            label_names if label_names else {i: str(i) for i in range(len(set(labels)))}
        )

        self.transform = transform if transform else None

        self.model = None  # tSNE or PCA model
        self.features = None

    def feature_extraction_from_path(self):
        """method for feature extraction."""
        self.features = []

        self.feature_extractor.eval()

        for image, label in zip(self.images, self.labels):
            image = read_image(image)

            image = self.transform(image)
            feature = self.feature_extractor(image.unsqueeze(0))
            self.features.extend(feature.detach().numpy())

    def feature_extraction_from_loader(self):
        self.features = []
        self.feature_extractor.eval()

        for image, label in zip(self.images, self.labels):
            feature = self.feature_extractor(image)
            self.features.extend(feature.cpu().numpy())

    def train_model(self):
        pass

    def save_graph(self):
        pass
