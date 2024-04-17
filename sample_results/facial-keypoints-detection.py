import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
train_df = pd.read_csv("./input/training/training.csv")
train_df.dropna(inplace=True)  # Remove missing values for simplicity

# Preprocess the data
X = (
    np.vstack(train_df["Image"].apply(lambda x: np.fromstring(x, sep=" ")).values)
    / 255.0
)  # Normalize pixel values
X = X.reshape(-1, 96, 96, 1)
y = train_df.drop(["Image"], axis=1).values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define dataset
class FacesDataset(Dataset):
    def __init__(self, images, keypoints):
        self.images = images
        self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)
        keypoint = torch.tensor(self.keypoints[idx], dtype=torch.float32)
        return image, keypoint


# Define model
class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 48 * 48, 1000)
        self.fc2 = nn.Linear(1000, 30)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training
def train(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, keypoints in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, keypoints in val_loader:
                outputs = model(images)
                loss = criterion(outputs, keypoints)
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}"
        )


# Initialize dataset, model, criterion, and optimizer
train_dataset = FacesDataset(X_train, y_train)
val_dataset = FacesDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
model = KeypointModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, criterion, optimizer, train_loader, val_loader, epochs=10)

# Evaluation
model.eval()
predictions = []
ground_truths = []
with torch.no_grad():
    for images, keypoints in val_loader:
        outputs = model(images)
        predictions.extend(outputs.numpy())
        ground_truths.extend(keypoints.numpy())

rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
print(f"Validation RMSE: {rmse}")
