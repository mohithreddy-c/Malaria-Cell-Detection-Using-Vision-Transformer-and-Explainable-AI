import torch
import torch.nn as nn
import timm
import time

import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
from PIL import Image
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2

# ----------------------------
# Image Preprocessing
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# ----------------------------
# Load Dataset
# ----------------------------

dataset = datasets.ImageFolder("dataset/cell_images", transform=transform)

print("Classes:", dataset.classes)
print("Total Images:", len(dataset))


# ----------------------------
# Use smaller subset (5000 images)
# ----------------------------
indices = torch.randperm(len(dataset))[:5000]
dataset = torch.utils.data.Subset(dataset, indices)


# ----------------------------
# Train / Validation Split
# ----------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])


# ----------------------------
# DataLoaders
# ----------------------------

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=0)
val_loader = DataLoader(val_dataset,batch_size=16,shuffle=False,num_workers=0)


# ----------------------------
# Load Vision Transformer
# ----------------------------

model = timm.create_model('vit_small_patch16_224', pretrained=True)

model.head = nn.Linear(model.head.in_features,2)


# Freeze layers (faster training)

for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True


# ----------------------------
# Device
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Device:",device)


# ----------------------------
# Loss and Optimizer
# ----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.0001)


# ----------------------------
# Training
# ----------------------------

epochs = 1
start = time.time()

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images,labels in tqdm(train_loader, desc="Training"):

        images,labels = images.to(device),labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Epoch",epoch+1,"Loss:",running_loss/len(train_loader))

print("Training time:",(time.time()-start)/60,"minutes")


# ----------------------------
# Validation Accuracy
# ----------------------------

model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for images,labels in val_loader:

        images,labels = images.to(device),labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = 100 * correct / total

print("Validation Accuracy:",accuracy)




# ----------------------------
# Confusion Matrix
# ----------------------------


cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Parasitized','Uninfected'],
            yticklabels=['Parasitized','Uninfected'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ----------------------------
# Predict Single Image
# ----------------------------

import torch.nn.functional as F

def predict_image(img_path):

    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        outputs = model(image_tensor)

        # convert logits to probabilities
        probs = F.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

    classes = ['Parasitized','Uninfected']

    prediction = classes[pred.item()]
    confidence_score = confidence.item()*100

    print("\nPrediction:", prediction)
    print("Confidence: {:.2f}%".format(confidence_score))

    # show image with prediction
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.title(f"{prediction} ({confidence_score:.2f}%)")
    plt.axis("off")
    plt.savefig("prediction_result.png")

    # probability bar chart
    plt.figure(figsize=(5,3))
    plt.bar(classes, probs.cpu().numpy()[0])
    plt.title("Prediction Confidence")
    plt.ylabel("Probability")
    plt.savefig("confidence_chart.png")

    print("\nSaved:")
    print("prediction_result.png")
    print("confidence_chart.png")


def reshape_transform(tensor, height=14, width=14):
    # remove class token
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # move channels to first dimension
    result = result.permute(0, 3, 1, 2)
    return result


def gradcam_visualization(img_path):

    image = Image.open(img_path).convert("RGB")
    rgb_img = np.array(image.resize((224,224))) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(device)

    # correct target layer for ViT
    target_layer = model.blocks[-1].norm1

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.savefig("gradcam_result.png")

    print("GradCAM image saved as gradcam_result.png")

# ----------------------------
# Simple Visualization
# ----------------------------

def show_image(img_path):

    image = Image.open(img_path).convert("RGB")
    img = transform(image)

    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.title("Input Cell Image")
    plt.axis('off')
    plt.show()


# Example prediction

print("Running prediction...")
predict_image("dataset/cell_images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png")


# Example visualization

print("Showing image...")
show_image("dataset/cell_images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png")

for param in model.parameters():
    param.requires_grad = True

gradcam_visualization("dataset/cell_images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png")