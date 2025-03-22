import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NiftiDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = self.get_all_nifti_files(image_dir)

    def get_all_nifti_files(self, folder_path):
        # Recusrievly find files in folder_path that match regex
        nii_files = []
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".nii"):
                    nii_files.append(os.path.join(subdir, file))

        return nii_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = nib.load(self.images[idx]).get_fdata()  # Load NIfTI image as NumPy array
        img = np.array(img, dtype=np.float32)
        
        # Normalize the image intensity
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Min-max normalization
        
        # Convert to 3D single-channel format (ResNet expects 3-channel input)
        img = np.stack(img * 3, axis=-1)  # Convert grayscale to RGB
        img = np.transpose(img, (2, 1, 0)) # Change to (C, H, W)

        if self.transform:
            img = self.transform(torch.tensor(img))

        label = self.get_label_from_filename(self.images[idx])  # Implement a function to map filenames to labels
        return img, label

    def get_label_from_filename(self, filename):
        """Extract label based on filename convention."""
        if "CN" in filename:
            return 0  # Cognitively Normal
        elif "MCI" in filename:
            return 1  # Mild Cognitive Impairment
        elif "AD" in filename:
            return 2  # Alzheimer's Disease
        return -1  # Unknown label

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit ResNet-50
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizing single-channel data
])

# Load datasets
print("Reading NiftiDataset")
train_dataset = NiftiDataset(image_dir=".\\adni_dataset\\ADNI", transform=transform)
val_dataset = NiftiDataset(image_dir=".\\adni_dataset\\ADNI", transform=transform)

print("Loading dataset with DataLoader")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained ResNet-50 model
class AlzheimerResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlzheimerResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Modify the output layer

    def forward(self, x):
        return self.model(x)

# Define model
model = AlzheimerResNet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Save the model
torch.save(model.state_dict(), "alzheimer_resnet50.pth")


