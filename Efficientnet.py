"""
Solar Panel Fault Classification — EfficientNet-B0 (99%
Target)
"""

import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0,
EfficientNet_B0_Weights

14

Transfer learning Comparative analysis Deep Learning – IA 2

from sklearn.metrics import accuracy_score,
classification_report, confusion_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = "./faulty_solar_panel/"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available()
else "cpu")

print("Device:", DEVICE)

train_tf = transforms.Compose([
transforms.Resize((IMG_SIZE, IMG_SIZE)),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(25),
transforms.ColorJitter(brightness=0.3, contrast=0.3,
saturation=0.3),
transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
transforms.RandomAffine(10, shear=5),
transforms.ToTensor(),

transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

15

Transfer learning Comparative analysis Deep Learning – IA 2

val_tf = transforms.Compose([
transforms.Resize((IMG_SIZE, IMG_SIZE)),
transforms.ToTensor(),

transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,
"train"), train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR,
"val"), val_tf)
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR,
"test"), val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
shuffle=False)

NUM_CLASSES = len(train_ds.classes)
print("Classes:", train_ds.classes)

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(

16

Transfer learning Comparative analysis Deep Learning – IA 2

nn.Dropout(0.5),
nn.Linear(in_features, 512),
nn.BatchNorm1d(512),
nn.ReLU(),
nn.Dropout(0.4),
nn.Linear(512, NUM_CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR,
weight_decay=1e-4)

scheduler =
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
T_max=EPOCHS)

best_acc = 0
train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []

for epoch in range(EPOCHS):

model.train()
train_loss, correct, total = 0, 0, 0

for imgs, labels in tqdm(train_loader):
imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

optimizer.zero_grad()

17

Transfer learning Comparative analysis Deep Learning – IA 2

outputs = model(imgs)
loss = criterion(outputs, labels)

loss.backward()
optimizer.step()

train_loss += loss.item()
preds = outputs.argmax(1)
correct += (preds == labels).sum().item()
total += labels.size(0)

train_acc = correct / total
train_loss /= len(train_loader)

model.eval()
val_preds, val_labels = [], []
val_loss = 0

with torch.no_grad():
for imgs, labels in val_loader:
imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
outputs = model(imgs)

loss = criterion(outputs, labels)
val_loss += loss.item()

preds = outputs.argmax(1).cpu().numpy()
val_preds.extend(preds)
val_labels.extend(labels.cpu().numpy())

val_acc = accuracy_score(val_labels, val_preds)

18

Transfer learning Comparative analysis Deep Learning – IA 2

val_loss /= len(val_loader)

scheduler.step()

train_acc_hist.append(train_acc)
val_acc_hist.append(val_acc)
train_loss_hist.append(train_loss)
val_loss_hist.append(val_loss)

print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val
Acc={val_acc:.4f}")

if val_acc > best_acc:
best_acc = val_acc
torch.save(model.state_dict(), "best_model.pth")

plt.figure()
plt.plot(train_acc_hist, label="Train Acc")
plt.plot(val_acc_hist, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

plt.figure()
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

19

Transfer learning Comparative analysis Deep Learning – IA 2

preds, labels_all = [], []

with torch.no_grad():
for imgs, labels in test_loader:
imgs = imgs.to(DEVICE)
outputs = model(imgs)
preds.extend(outputs.argmax(1).cpu().numpy())
labels_all.extend(labels.numpy())

print("\n Test Accuracy:", accuracy_score(labels_all, preds))
print("\nReport:\n", classification_report(labels_all, preds,
target_names=train_ds.classes))

cm = confusion_matrix(labels_all, preds)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()