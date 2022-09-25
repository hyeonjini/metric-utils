import torch
import numpy as np
import matplotlib.cm as cm

class GradientCAM():

    fmap_pool = {}
    grad_pool = {}
    handlers = []
    
    def __init__(
        self,
        target_image = None,
        target_layer: str = None,
        topk: int = 3,
        cnn_model: torch.nn.Module = None,
        transform: torch.uitls.transform = None,
        paper_cmap: bool =True
    ):
        self.target_image = target_image
        self.target_layer = target_layer
        self.topk = topk
        self.cnn_model = cnn_model
        self.transform = transform
        self.paper_cmap = paper_cmap

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.register_hook()
    
    def __call__(self):
        
        gcam_list = []
        org_image = self.target_image
        image = self.transform(org_image)

        # generate grad-cam
        self.cnn_model.to(self.device).eval()

        with torch.no_grad():
            image = torch.tensor(image).to(self.device)
            image = image.unsqueeze(0)
            probs, ids = self.forward(image)
            print(probs, ids)
            print(f"Generate Grad-CAM {self.target_layer}")

            for i in range(self.topk):
                self.backward(ids=ids[:, [i]])
                regions = self.generate(target_layer=self.target_layer)
                gcam = regions[0, 0]
                
                if self.device == 'cuda':
                    gcam = gcam.cpu().numpy()
                else:
                    gcam = gcam.numpy()
                
                cmap = cm.jet_r(gcam)[..., :3] * 255.0
                cmap = cmap[..., ::-1].copy()
                if self.paper_cmap:
                    alpha = gcam[..., None]
                    gcam = alpha * cmap + (1 - alpha) * org_image
                else:
                    gcam = (cmap.astype(np.float) + org_image.astype(np.float))

                gcam_list.append([np.uint8(gcam)])

        self.remove_hook()

        return gcam_list
    
    def register_hook(self):
        """ register hooks
        """
        
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()
            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()
                
            return backward_hook
        
        for name, module in self.model.named_modules():

            if self.target_layer is None or name == self.target_layer:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError(f"There is no name in pool: {target_layer}")
    
    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.cnn_model(image)
        self.probs = torch.nn.functional.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = torch.nn.functional.adaptive_avg_pool2d(grads, 1)
        
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)

        gcam = torch.functional.relu(gcam)
        gcam = torch.functional.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    def image_to_tensor(self, raw_image):
        image = raw_image

        return image        
    
        