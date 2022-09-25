import torch


class GradientCAM():
    
    def __init__(
        self,
        target_image = None,
        target_layer: str = None,
        cnn_model: torch.nn.Module = None,
        transform: torch.uitls.transform = None,
    ):
        pass

    
    def register_hook(self):
        pass


    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()
        