import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import build_model
import os


DATA_DIR = "data/brain_tumor_dataset/"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
SAVE_PATH = "models/efficientnet_b0.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
    ]
)

validate_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
    ]
)


def run_epoch(phase, opt):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    loader = train_dl if is_train else validate_dl
    total_loss, correct = 0, 0
    
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if is_train: 
                opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if is_train:
                loss.backward()
                opt.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()

    n = len(train_ds) if is_train else len(validate_ds)
    return total_loss / n, correct / n


full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
n_val = int(len(full_ds) * 0.2)
train_ds, validate_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
validate_ds.dataset.transform = validate_tf

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
validate_dl = DataLoader(validate_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"Classes : {full_ds.classes}") 
print(f"Train: {len(train_ds)} | Validate: {len(validate_ds)} | Device: {DEVICE}")


model = build_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
os.makedirs("models", exist_ok=True)
best_accuracy = 0.0


# PHASE 1 - train only the head 
print("----- PHASE 1: Train Head Only -----")
optimizer1 = torch.optim.Adam(model.classifier.parameters(), lr=LR)

for epoch in range(1, 4):
    tl, ta = run_epoch("train", optimizer1)
    vt, va = run_epoch("validate", optimizer1) 
    print(f"Epoch {epoch:02d} | Loss: {tl:.4f} | Train: {ta:.3f} | Val: {va:.3f}")
    if va > best_accuracy:
        best_accuracy = va 
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved accuracy: {best_accuracy:.3f}")


# PHASE 2 - fine-tuning the whole model
print("----- PHASE 2: Fine-tuning the model -----")
for p in model.parameters():
    p.requires_grad = True
optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS - 3)

for epoch in range(4, EPOCHS + 1):
    tl, ta = run_epoch("train", optimizer2)
    vl, va = run_epoch("validate", optimizer2)
    print(f"Epoch {epoch:02d} | Loss: {tl:.4f} | Train: {ta:.3f} | Val: {va:.3f}")
    if va > best_accuracy:
        best_accuracy = va 
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved accuracy: {best_accuracy:.3f}")

print(f"Overall best accuracy: {best_accuracy}")
