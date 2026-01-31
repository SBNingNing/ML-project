import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

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
TEST_DATA_ROOT = '/opt/data/private/ML-project/dataset/test'

def evaluate_model(model, test_loader, device):
    model.eval()
    
    tp_micro = 0
    fp_micro = 0
    fn_micro = 0
    
    # Thresholds matching For_TA_test.py
    # [No Defect, Chipped, Scratch, Stain]
    thresholds = torch.tensor([0.5, 0.5, 0.4, 0.3], device=device)
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            # labels are (B, 4) multi-hot
            labels = labels.to(device)
            
            out_cls, _ = model(images)
            probs = torch.sigmoid(out_cls)
            
            preds = (probs > thresholds).float()
            
            # Micro - calculation on all elements
            # TP: pred=1, label=1
            tp = (preds * labels).sum()
            # FP: pred=1, label=0
            fp = (preds * (1 - labels)).sum()
            # FN: pred=0, label=1
            fn = ((1 - preds) * labels).sum()
            
            tp_micro += tp.item()
            fp_micro += fp.item()
            fn_micro += fn.item()
            
    epsilon = 1e-15
    precision_micro = tp_micro / (tp_micro + fp_micro + epsilon)
    recall_micro = tp_micro / (tp_micro + fn_micro + epsilon)
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + epsilon)
    
    return precision_micro, recall_micro, f1_micro

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
    train_dataset = GlassDataset(root_dir=DATA_ROOT, transform=train_transform, augment=True)
    print(f"Train Dataset Size: {len(train_dataset)}")
    
    # "Do NOT use random_split. Use the full dataset for training."
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Test Dataset
    test_loader = None
    if os.path.exists(TEST_DATA_ROOT):
        test_dataset = GlassDataset(root_dir=TEST_DATA_ROOT, transform=train_transform, augment=False)
        print(f"Test Dataset Size: {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print(f"Warning: Test data root '{TEST_DATA_ROOT}' does not exist.")
    
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
    history = []
    
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
        epoch_loss = running_loss / len(train_dataset)
        epoch_cls_loss = running_cls_loss / len(train_dataset)
        epoch_seg_loss = running_seg_loss / len(train_dataset)
        
        # Evaluation on Test Set
        p_micro, r_micro, f1_micro = 0.0, 0.0, 0.0
        if test_loader is not None:
             p_micro, r_micro, f1_micro = evaluate_model(model, test_loader, DEVICE)
             print(f"Test Set - Micro P: {p_micro:.4f}, Micro R: {r_micro:.4f}, Micro F1: {f1_micro:.4f}")
             # Switch back to train mode
             model.train()
             
        history.append({
            'epoch': epoch + 1,
            'total_loss': epoch_loss,
            'micro_p': p_micro,
            'micro_r': r_micro,
            'micro_f1': f1_micro
        })

        # Save history to file
        with open('training_log.txt', 'w') as f:
            f.write("Epoch\tLoss\tMicroP\tMicroR\tMicroF1\n")
            for entry in history:
                f.write(f"{entry['epoch']}\t{entry['total_loss']:.4f}\t{entry['micro_p']:.4f}\t{entry['micro_r']:.4f}\t{entry['micro_f1']:.4f}\n")
        
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
