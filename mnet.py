import os, warnings, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = "./Faulty_solar_panels_split/"
IMG_SIZE = 192          # ⚡ faster than 224
BATCH_SIZE = 32
EPOCHS = 12             # ⛔ no overtraining
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# ─────────────────────────────────────────────
# TRANSFORMS (FAST + EFFECTIVE)
# ─────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), val_tf)

NUM_CLASSES = len(train_ds.classes)
print("Classes:", train_ds.classes)

# 🔥 CLASS BALANCING
class_counts = np.bincount(train_ds.targets)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_ds.targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights)

# Freeze backbone
for p in model.features.parameters():
    p.requires_grad = False

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)

# ─────────────────────────────────────────────
# LOSS + OPTIMIZER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(
    weight=torch.FloatTensor(class_weights).to(DEVICE)
)

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
best_acc = 0
patience, counter = 3, 0   # 🔥 early stopping

for epoch in range(EPOCHS):

    # 🔥 Controlled fine-tuning
    if epoch == 3:
        for p in model.features[-2:].parameters():
            p.requires_grad = True
        print("→ Fine-tuning started")

    model.train()
    correct, total = 0, 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # VALIDATION
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds_all.extend(outputs.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())

    val_acc = accuracy_score(labels_all, preds_all)
    scheduler.step()

    print(f"Epoch {epoch+1} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

    # Early stopping logic
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_mobilenetv2_final.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(" Early stopping triggered")
            break

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("best_mobilenetv2_final.pth"))
model.eval()

preds, labels_all = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds.extend(outputs.argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())

print("\n FINAL Test Accuracy:", accuracy_score(labels_all, preds))
print("\nReport:\n", classification_report(labels_all, preds, target_names=train_ds.classes))