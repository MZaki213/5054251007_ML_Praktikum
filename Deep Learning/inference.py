import torch
from model_def import CNN

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_model(path="model.pth"):
    model = CNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_probs(model, x):
    """x: numpy float32 berbentuk (1, 1, 28, 28), nilai 0-1."""
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        return torch.softmax(logits, dim=1)[0].numpy()