import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import GlassDataset
from model_mtl import MultiTaskResNet

# Configuration
USE_AUX = True
SEG_LOSS_WEIGHT = 1.0
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use absolute path for robustness or relative path from this script's location
DATA_ROOT = '/opt/data/private/ML-project/dataset/train'

def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. Data
    # Define transforms
    # ResNet models typically expect normalization with ImageNet mean/std
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Check if data directory exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root directory '{DATA_ROOT}' does not exist.")
        return

    print("Loading Dataset...")
    dataset = GlassDataset(root_dir=DATA_ROOT, transform=train_transform)
    print(f"Dataset Size: {len(dataset)}")
    
    # "Do NOT use random_split. Use the full dataset for training."
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Model
    print(f"Initializing Model (Auxiliary Head: {USE_AUX})...")
    model = MultiTaskResNet(use_aux=USE_AUX)
    model.to(DEVICE)
    
    # 3. Loss
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_seg = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_loss = float('inf')
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for images, labels, masks in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Forward pass
            out_cls, out_seg = model(images)
            
            # Compute Cls Loss
            loss_cls = criterion_cls(out_cls, labels)
            
            # Compute Seg Loss (Conditional)
            if USE_AUX and out_seg is not None:
                loss_seg = criterion_seg(out_seg, masks)
                total_loss = loss_cls + SEG_LOSS_WEIGHT * loss_seg
            else:
                loss_seg = torch.tensor(0.0, device=DEVICE)
                total_loss = loss_cls
                
            # Backward & Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Stats
            batch_size = images.size(0)
            running_loss += total_loss.item() * batch_size
            running_cls_loss += loss_cls.item() * batch_size
            running_seg_loss += loss_seg.item() * batch_size
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}", 
                'Cls': f"{loss_cls.item():.4f}", 
                'Seg': f"{loss_seg.item():.4f}"
            })
            
        # Step Scheduler
        scheduler.step()
        
        # Calculate epoch average losses
        epoch_loss = running_loss / len(dataset)
        epoch_cls_loss = running_cls_loss / len(dataset)
        epoch_seg_loss = running_seg_loss / len(dataset)
        
        print(f"Epoch {epoch+1} Summary: Total Loss: {epoch_loss:.4f} | Cls Loss: {epoch_cls_loss:.4f} | Seg Loss: {epoch_seg_loss:.4f}")
        
        # 5. Saving
        # Save Best Model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best_model.pth (New Best Loss: {best_loss:.4f})")
            
    # Save Last Model
    torch.save(model.state_dict(), 'last_model.pth')
    print("Saved last_model.pth")
    print("Training Complete.")

if __name__ == '__main__':
    main()
