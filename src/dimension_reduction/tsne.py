from src.dimension_reduction import DimensionReduction
from src.utils import scale_to_01_range

import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE


class tSNE(DimensionReduction):
    def __init__(self, **kwargs) -> None:
        super(tSNE, self).__init__(**kwargs)

    def train_model(self, n_components: int = 2, _from: str = "path"):
        if _from == "path":
            self.feature_extraction_from_path()
        else:
            self.feature_extraction_from_loader()

        self.model = TSNE(n_components=n_components).fit_transform(self.features)

    def visualize_tsne_with_point(self, figsize=(12, 8)):

        tx = self.model[:, 0]
        ty = self.model[:, 1]

        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
            for i in range(len(self.label_names))
        ]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for label in tqdm(range(min(self.labels), max(self.labels) + 1)):
            indices = [i for i, l in enumerate(self.labels) if l == label]

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            ax.scatter(
                current_tx,
                current_ty,
                c=colors[label],
                label=f"{self.label_names[label]}",
                alpha=0.7,
            )

        ax.legend(loc="best")
        plt.show()

    def visualize_tsne_with_image(self):
        assert self.model != None

    def visualize_tsne_point(
        self, tx, ty, labels, figsize=(12, 8), colors_per_class=None
    ):
        assert self.model != None
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

    def save_graph(self, save_pth: str):
        pass
