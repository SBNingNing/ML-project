import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class GlassDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'dataset/train').
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' or 'test'.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Get all image files
        self.img_paths = glob.glob(os.path.join(root_dir, 'img', '*.png'))
        # Sort to ensure consistency
        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 320x320 (Native resolution)
        image = cv2.resize(image, (320, 320))
        
        # Normalize to [0, 1] and convert to tensor (C, H, W)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        if self.transform:
            image = self.transform(image)

        # If test mode, we might not have labels/masks, but the prompt implies 
        # this dataset class is used for training where we need targets.
        # For inference (For_TA_test.py), we might use a simplified version or just ignore targets.
        # However, let's implement target generation logic.
        
        # Construct label path
        # Image: .../img/filename.png -> Label: .../txt/filename.txt
        filename = os.path.basename(img_path)
        txt_name = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(self.root_dir, 'txt', txt_name)
        
        # Initialize targets
        # labels: [No Defect, Chipped, Scratch, Stain]
        labels = torch.zeros(4, dtype=torch.float32)
        # mask: (1, 320, 320)
        mask = torch.zeros((1, 320, 320), dtype=torch.float32)
        
        has_defect = False
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    has_defect = True
                    class_id = int(parts[0]) # 0: Chipped, 1: Scratch, 2: Stain
                    x_c, y_c, w, h = map(float, parts[1:])
                    
                    # Map class_id to internal index
                    # 0(Chipped) -> 1
                    # 1(Scratch) -> 2
                    # 2(Stain)   -> 3
                    internal_idx = class_id + 1
                    if internal_idx < 4:
                        labels[internal_idx] = 1.0
                    
                    # Generate mask
                    # Convert normalized coords to 320x320
                    # x_c, y_c, w, h are normalized [0, 1]
                    
                    img_h, img_w = 320, 320
                    
                    x_center = x_c * img_w
                    y_center = y_c * img_h
                    width = w * img_w
                    height = h * img_h
                    
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    
                    # Clip to image boundaries
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_w, x_max)
                    y_max = min(img_h, y_max)
                    
                    mask[0, y_min:y_max, x_min:x_max] = 1.0

        if not has_defect:
            labels[0] = 1.0
            
        return image, labels, mask
