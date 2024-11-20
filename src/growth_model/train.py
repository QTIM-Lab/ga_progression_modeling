import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import wandb
from sklearn.model_selection import train_test_split

# Load CSV
csv_path = 'data/GA_progression_modelling/model_training_dataset/data.csv'
data_df = pd.read_csv(csv_path)

# Split dataset into train and validation sets
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

class GrowthRateDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'file_path']
        image = Image.open(img_name).convert("RGB")
        growth_rate = self.data.loc[idx, 'growth_rate']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(growth_rate, dtype=torch.float)

# Transformations for the images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = GrowthRateDataset(dataframe=train_df, transform=transform)
val_dataset = GrowthRateDataset(dataframe=val_df, transform=transform)

# Define a simple CNN model for regression
class Model(nn.Module):
    def __init__(self, model_name='resnet18'):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, in_chans=1, pretrained=True, num_classes=1)

    def forward(self, x):
        return self.model(x)

def train():
    # Initialize Weights and Biases
    wandb.init(project="growth-rate-regression")

    # Retrieve hyperparameters
    config = wandb.config

    # load model, loss and optimizer
    model = Model(model_name=config.model_name)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # Training loop
    num_epochs = config.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss = val_loss / len(val_dataloader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log the loss to Weights and Biases
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss, "val_loss": val_loss})

    # Save the model
    model_save_path = os.path.join(wandb.run.dir, f'growth_rate_model.pth')
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':

    wandb.login()

    # define wandb sweep config
    sweep_configuration = {
        'method': 'grid',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'num_epochs': {
                'values': [1000, 2000, 3000, 4000, 5000]
            },
            'learning_rate': {
                'values': [0.001, 0.0001, 0.00001]
            },
            'model_name': {
                'values': ['resnet18', 'resnet34', 'resnet50']
            }
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='growth-rate-regression')

    # Run the sweep
    wandb.agent(sweep_id, function=train, count=20)