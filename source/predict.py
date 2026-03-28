#----- MODEL TESTING -----
import sys

import torch
from torchvision import transforms
from PIL import Image
from model import build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/efficientnet_b0.pth"
CLASSES = ["No Tumor", "Tumor Detected"]


def predict(img_path: str):
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        pred = probs.argmax().item()

    print(f"\nImage : {img_path}")
    print(f"Result: {CLASSES[pred]}")
    print(f"  No Tumor : {probs[0].item()*100:.1f}%")
    print(f"  Tumor    : {probs[1].item()*100:.1f}%")


tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python source/predict.py <path/to/image.jpg>")
        sys.exit(1)
    predict(sys.argv[1])
