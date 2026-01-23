import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class GlassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with 'img' and 'txt' subdirectories.
            transform (callable, optional): Optional transform to be applied on the image (usually ToTensor + Normalize).
                                           Note: Resizing and geometric augmentations are handled internally
                                           to ensure image-mask consistency.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.img_paths = glob.glob(os.path.join(root_dir, 'img', '*.png'))
        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Determine txt path
        filename = os.path.basename(img_path)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(self.root_dir, 'txt', txt_filename)
        
        # Read Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Handle image loading errors
            print(f"Error loading {img_path}: {e}")
            # Return a dummy black image to avoid crashing
            image = Image.new('RGB', (320, 320))

        # 1. Resize Step (Target Resolution 320x320)
        target_h, target_w = 320, 320
        image = image.resize((target_w, target_h), Image.BILINEAR)
        
        # Initialize Labels
        # Labels: [No Defect, Chipped, Scratch, Stain]
        labels = torch.zeros(4, dtype=torch.float32)
        
        # Initialize Mask (1, 320, 320)
        mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        
        yolo_to_internal = {0: 1, 1: 2, 2: 3}
        has_defect = False
        
        # Generate Mask from TXT
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 0:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            
                            if cls_id in yolo_to_internal:
                                has_defect = True
                                internal_idx = yolo_to_internal[cls_id]
                                labels[internal_idx] = 1.0
                                
                                x_c, y_c, w, h = map(float, parts[1:])
                                
                                # Convert normalized coords to target_h/target_w
                                x1 = int((x_c - w / 2) * target_w)
                                y1 = int((y_c - h / 2) * target_h)
                                x2 = int((x_c + w / 2) * target_w)
                                y2 = int((y_c + h / 2) * target_h)
                                
                                x1 = max(0, min(target_w, x1))
                                y1 = max(0, min(target_h, y1))
                                x2 = max(0, min(target_w, x2))
                                y2 = max(0, min(target_h, y2))
                                
                                if x2 > x1 and y2 > y1:
                                    mask[0, y1:y2, x1:x2] = 1.0
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")

        if not has_defect:
            labels[0] = 1.0
            
        # 2. Data Augmentation (Synchronized for Image & Mask)
        # Apply transforms on PIL Image and Tensor Mask
        # Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # Random Vertical Flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # Random Rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 3. Apply Final Transforms (ToTensor, Normalize) to Image
        if self.transform:
            image = self.transform(image)
            
        return image, labels, mask
