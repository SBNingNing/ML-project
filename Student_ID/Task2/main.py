import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from dataset import GlassDataset
from model_mtl import MultiTaskResNet

def train():
    # Hyperparameters
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    # Assuming the script is run from Student_ID/Task2/
    # Dataset is at ../../dataset/train
    ROOT_DIR = os.path.join('..', '..', 'dataset', 'train')
    
    if not os.path.exists(ROOT_DIR):
        # Fallback or check absolute path if needed
        # For now, assume the structure provided in the prompt
        print(f"Warning: Dataset root {ROOT_DIR} does not exist. Please check path.")
    
    # Dataset and DataLoader
    train_dataset = GlassDataset(root_dir=ROOT_DIR, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Model
    model = MultiTaskResNet().to(DEVICE)
    
    # Loss Functions
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_seg = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    
    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (images, labels, masks) in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Forward pass
            out_cls, out_seg = model(images)
            
            # Calculate losses
            loss_cls = criterion_cls(out_cls, labels)
            loss_seg = criterion_seg(out_seg, masks)
            
            # Total loss
            total_loss = loss_cls + 1.0 * loss_seg
            
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_cls_loss += loss_cls.item()
            running_seg_loss += loss_seg.item()
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=total_loss.item())
            
        epoch_loss = running_loss / len(train_loader)
        epoch_cls_loss = running_cls_loss / len(train_loader)
        epoch_seg_loss = running_seg_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.4f}, Seg: {epoch_seg_loss:.4f})")
        
        # Save best model based on total loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best_model.pth")

    print("Training complete.")

if __name__ == '__main__':
    train()
