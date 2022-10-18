from dimension_reduction import DimensionReduction

from typing import Optional, List, Dict, Any
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.manifold import TSNE


class tSNE(DimensionReduction):
    def __init__(
        self,
        feature_extractor: Module,
        images: List,
        labels: List,
        label_names: Optional[Dict[Any, Any]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        super.__init__()

    def train_model(self, n_components: int = 2):
        self.feature_extraction()
        self.model = TSNE(n_components=n_components)
        self.model.fit_transform(self.features)

    def save_graph(self, save_pth: str):
        pass

    def visualize_tsne_point(
        self, tx, ty, labels, figsize=(12, 8), colors_per_class=None
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        print(min(labels), max(labels))
        for label in tqdm(range(min(labels), max(labels) + 1)):
            indices = [i for i, l in enumerate(labels) if l == label]

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

            ax.scatter(
                current_tx,
                current_ty,
                c=colors_per_class[label - min(labels)],
                label=str(self.label_names[label]),
                alpha=0.7,
            )

        ax.legend(loc="best")
        plt.show()
