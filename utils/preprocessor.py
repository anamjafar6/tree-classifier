import numpy as np
import torch
from torchvision import transforms
from PIL import Image

PYTORCH_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_for_pytorch(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = PYTORCH_TRANSFORM(image)
    return tensor.unsqueeze(0)

def preprocess_for_keras(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)
