import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import tensorflow as tf

class PyTorchGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_feature_maps)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, tensor):
        tensor = tensor.clone().requires_grad_(True)
        output = self.model(tensor)
        predicted_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, predicted_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def _overlay_heatmap(original_image, cam):
    img_resized = np.array(original_image.convert("RGB").resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

def gradcam_pytorch(original_image, model, tensor, target_layer):
    gc = PyTorchGradCAM(model, target_layer)
    try:
        cam = gc.compute(tensor)
    finally:
        gc.remove_hooks()
    return _overlay_heatmap(original_image, cam)

def gradcam_keras_manual(original_image, model, array, predicted_class):
    img_tensor = tf.cast(array, tf.float32)

    # Sequential model ke andar MobileNetV2 base model hai
    base_model = model.layers[0]  # mobilenetv2_1.00_224

    # Base model ki last conv layer
    last_conv_layer = None
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer

    if last_conv_layer is None:
        return original_image.resize((224, 224))

    # Base model input se last conv output tak grad model
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_out = grad_model(img_tensor, training=False)
        # Baaki layers manually apply karo
        x = base_out
        for layer in model.layers[1:]:
            x = layer(x, training=False)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    cam = conv_out @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    cam = cam.numpy()

    return _overlay_heatmap(original_image, cam)

def generate_gradcam_pytorch_resnet(original_image, model, tensor):
    return gradcam_pytorch(original_image, model, tensor, model.layer4)

def generate_gradcam_pytorch_mobilenet(original_image, model, tensor):
    return gradcam_pytorch(original_image, model, tensor, model.features[-1])

def generate_gradcam_keras(original_image, model, array, predicted_class):
    return gradcam_keras_manual(original_image, model, array, predicted_class)
