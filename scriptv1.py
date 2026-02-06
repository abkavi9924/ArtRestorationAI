import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.checkpoint import checkpoint
import os
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
import time
import signal

stop_training = False

def signal_handler(signum, frame):
    global stop_training
    print("Gracefully stopping the training after this batch...")
    stop_training = True


# signal.signal(signal.SIGINT, signal_handler) 'COmmented because of problems in saved .pth file'

checkpoint_dir = r'C:\Users\Abdul Kavi Chaudhary\Desktop\Checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_spatial.pth")

# Function to save the model's state
def save_checkpoint(epoch, model, optimizer, scaler, scheduler, loss, map_location=None):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    if map_location:
        torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

# Function to load the model's state from a checkpoint
def load_checkpoint(model, optimizer, scaler, scheduler):
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch + 1}")
        return start_epoch, checkpoint['loss']
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, None

# trainset class
class ArtRestorationtrainset(Dataset):
    def __init__(self, clear_dir, distorted_dir, transform=None):
        self.clear_images = sorted([os.path.join(clear_dir, f) for f in os.listdir(clear_dir)])
        self.distorted_images = sorted([os.path.join(distorted_dir, f) for f in os.listdir(distorted_dir)])
        self.transform = transform
        self.distorted_per_clear = 5

    def __len__(self):
        return len(self.distorted_images)

    def __getitem__(self, idx):
        clear_idx = idx // self.distorted_per_clear
        clear_img = cv2.imread(self.clear_images[clear_idx])
        distorted_img = cv2.imread(self.distorted_images[idx])

        if self.transform:
            clear_img = self.transform(clear_img)
            distorted_img = self.transform(distorted_img)

        return distorted_img, clear_img

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, checkpoint_path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.checkpoint_path)
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")


# Training configuration
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Load trainset
# Split trainset into training and validation
trainset = ArtRestorationtrainset(
    clear_dir=r'C:\Users\Abdul Kavi Chaudhary\Desktop\spatial_ds\clear',
    distorted_dir=r'C:\Users\Abdul Kavi Chaudhary\Desktop\spatial_ds\distorted',
    transform=transform
)

validation_split = 0.2
train_size = int((1 - validation_split) * len(trainset))
val_size = len(trainset) - train_size
train_trainset, val_trainset = random_split(trainset, [train_size, val_size])

train_loader = DataLoader(train_trainset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_trainset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RestorationUNet().to(device)
pixelwise_loss = nn.MSELoss()
perceptual_loss = PerceptualLoss(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # For mixed precision
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Learning rate scheduler

start_epoch, previous_loss = load_checkpoint(model, optimizer, scaler, scheduler)
early_stopping = EarlyStopping(patience=5, verbose=True)

def save_model(model, path):
    # Move the model to CPU
    model_cpu = model.to(torch.device('cpu'))
    torch.save(model_cpu.state_dict(), path)
    print(f"Model saved to {path}")
    
def calculate_accuracy(restored, clear, threshold=0.1):
    # Assuming pixel-wise accuracy based on a threshold
    diff = torch.abs(restored - clear)
    correct = (diff < threshold).float()  # Count as correct if the difference is below threshold
    accuracy = correct.mean().item() * 100  # Accuracy as percentage
    return accuracy

from tqdm import tqdm
    
if __name__ == "__main__":
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        if stop_training:
            print(f"Training stopped at epoch {epoch}. Please restart after checkpointing.")
            save_checkpoint(epoch, model, optimizer, scaler, scheduler, loss)
            break
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0 
        running_accuracy =0
        for batch_idx, (distorted, clear) in enumerate(train_loader):

            optimizer.zero_grad()

            with autocast("cuda"):
                distorted = distorted.to(device)
                clear = clear.to(device)
                restored = model(distorted)
                loss1 = pixelwise_loss(restored, clear)
                loss2 = perceptual_loss(restored, clear) if batch_idx % 5 == 0 else torch.tensor(0.0, device=device)
                loss = loss1 + 0.1 * loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
            batch_accuracy = calculate_accuracy(restored, clear)
            running_accuracy += batch_accuracy
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}],Accuracy = {batch_accuracy:.4f}, Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {avg_train_accuracy:.2f}")
        
        model.eval()
        val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for distorted, clear in val_loader:
                distorted = distorted.to(device)
                clear = clear.to(device)
                restored = model(distorted)
                loss1 = pixelwise_loss(restored, clear)
                loss2 = perceptual_loss(restored, clear)
                loss = loss1 + 0.1 * loss2
                val_loss += loss.item()
                

                # Calculate pixel accuracy (optional)
                pred_binary = (restored > 0.5).float()  # Example threshold for binary pixel accuracy
                clear_binary = (clear > 0.5).float()
                correct_pixels += (pred_binary == clear_binary).sum().item()
                total_pixels += clear.numel()
                val_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_pixels / total_pixels * 100  # in percentage
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Check early stopping criteria
        early_stopping(epoch_loss, model)

        # Check if early stopping condition is met
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        save_checkpoint(epoch, model, optimizer, scaler, scheduler, epoch_loss)

        epoch_end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_end_time - epoch_start_time:.2f} seconds")



    save_model(model, 'restoration_unet.pth')
    
    
    
    
    



