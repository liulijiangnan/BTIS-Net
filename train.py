import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from dataset import BRATSDataset
from model import BTISNet

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# Parameters
data_dir = '/kaggle/working/brats-2021-task1/BraTS2021_Training_Data'
batch_size = 8
val_split = 0.2  # Proportion of the dataset to include in the validation set
test_split = 0.1  # Proportion of the dataset to include in the test set
shuffle_dataset = True  # Shuffle dataset before splitting
random_seed = 42  # Random seed for reproducibility

# Load the dataset
dataset = BRATSDataset(data_dir)

# Determine the lengths of training, validation, and test sets
dataset_size = len(dataset)
test_size = int(test_split * dataset_size)
val_size = int(val_split * (dataset_size - test_size))
train_size = dataset_size - val_size - test_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed)
)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Optionally print the sizes of each dataset to verify
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

model = BTISNet(in_channels=1, out_channels=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = DiceLoss()
scaler = GradScaler()

# Early stopping parameters
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0

# Number of epochs
num_epochs = 100

os.makedirs('model_weights', exist_ok=True)

checkpoint_path = 'model_weights/last_checkpoint_t1.pth'
best_model_path = 'model_weights/best_model_t1.pth'

# Load checkpoint if it exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming training from epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loader = tqdm(train_dataloader, desc="Training", leave=False)
    for vimage, mask in train_loader:
        vimage, mask = vimage.to(device), mask.to(device)
        optimizer.zero_grad()
        with autocast():
            predicted_mask, _, _, _ = model(vimage)
            loss = loss_fn(predicted_mask, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        with autocast():
            validation_loader = tqdm(val_dataloader, desc="Validation", leave=False)
            for vimage, mask in validation_loader:
                vimage, mask = vimage.to(device), mask.to(device)
                predicted_mask, _, _, _ = model(vimage)
                loss = loss_fn(predicted_mask, mask)
                val_loss += loss.item()

    val_loss /= len(val_dataloader)
    print(f"\rEpoch {epoch+1}, Validation Loss: {val_loss}", end='', flush=True)

    # Save the checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, checkpoint_path)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(os.listdir('model_weights'))
