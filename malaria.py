import torch
import torch.nn as nn
import timm
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --------------------------
# Image Preprocessing
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# --------------------------
# Load Dataset
# --------------------------
dataset = datasets.ImageFolder("dataset", transform=transform)

print("Classes:", dataset.classes)
print("Total Images:", len(dataset))

# --------------------------
# Train Validation Split
# --------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

# --------------------------
# DataLoaders
# --------------------------
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset,batch_size=16,shuffle=False,num_workers=4)

# --------------------------
# Load Vision Transformer
# --------------------------
model = timm.create_model('vit_small_patch16_224', pretrained=True)

model.head = nn.Linear(model.head.in_features,2)

# Freeze layers for faster training
for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Device:",device)

# --------------------------
# Loss and Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.0001)

# --------------------------
# Training
# --------------------------
epochs = 1
start = time.time()

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images,labels in train_loader:

        images,labels = images.to(device),labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Epoch",epoch+1,"Loss:",running_loss/len(train_loader))

print("Training time:",(time.time()-start)/60,"minutes")

# --------------------------
# Evaluation
# --------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images,labels in val_loader:

        images,labels = images.to(device),labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Validation Accuracy:",accuracy)