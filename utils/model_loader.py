import os
import torch
import torch.nn as nn
from torchvision import models as torchvision_models
import tensorflow as tf

def build_tree_vs_nontree():
    model = torchvision_models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model

def build_mango_stage():
    model = torchvision_models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 4)
    )
    return model

def build_gum_stage():
    model = torchvision_models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 4)
    )
    return model

def load_pytorch_model(build_fn, weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")
    model = build_fn()
    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_keras_model(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")
    return tf.keras.models.load_model(weights_path)

def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "tree_vs_nontree": load_pytorch_model(build_tree_vs_nontree, "models/tree_vs_nontree.pt", device),
        "species":         load_keras_model("models/best_mobilenetv2_model.h5"),
        "mango_stage":     load_pytorch_model(build_mango_stage, "models/mobilenetv2_tree_classifier.pth", device),
        "gum_stage":       load_pytorch_model(build_gum_stage, "models/gum_stage.pth", device),
        "device":          device
    }
