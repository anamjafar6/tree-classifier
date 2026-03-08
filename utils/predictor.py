import torch
import numpy as np
from PIL import Image
from utils.preprocessor import preprocess_for_pytorch, preprocess_for_keras

TREE_LABELS    = {0: "Non-Tree", 1: "Tree"}
SPECIES_LABELS = {0: "White Gum", 1: "Mango"}
STAGE_LABELS   = {0: "Seedling", 1: "Sapling", 2: "Mature", 3: "Overmature"}

def predict_tree(image, model, device):
    tensor = preprocess_for_pytorch(image).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return TREE_LABELS[predicted.item()], confidence.item(), tensor

def predict_species(image, model):
    array = preprocess_for_keras(image)
    raw = model.predict(array, verbose=0)[0][0]
    if raw > 0.5:
        return "Mango", float(raw), array
    else:
        return "White Gum", float(1.0 - raw), array

def predict_stage(image, model, device):
    tensor = preprocess_for_pytorch(image).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return STAGE_LABELS[predicted.item()], confidence.item(), tensor
