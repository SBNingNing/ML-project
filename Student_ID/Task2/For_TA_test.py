import os
import argparse
import json
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model_mtl import MultiTaskResNet

# Internal to Output Mapping
# Internal 0 (No Defect) -> Output "1"
# Internal 1 (Chipped)   -> Output "2"
# Internal 2 (Scratch)   -> Output "3"
# Internal 3 (Stain)     -> Output "4"
INDEX_TO_LABEL = {
    0: "1",
    1: "2",
    2: "3",
    3: "4"
}

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Determine where images are
        candidates = []
        # Check for img subdirectory
        img_subdir = os.path.join(root_dir, 'img')
        if os.path.exists(img_subdir) and os.path.isdir(img_subdir):
             candidates.extend([os.path.join(img_subdir, f) for f in os.listdir(img_subdir)])
        else:
             # Assume images are directly in root_dir
             candidates.extend([os.path.join(root_dir, f) for f in os.listdir(root_dir)])
             
        self.img_paths = [f for f in candidates if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
        except:
            # Return dummy if failed (black image)
            image = Image.new('RGB', (320, 320))
            
        if self.transform:
            image = self.transform(image)
        
        # Return filename for JSON key
        filename = os.path.basename(path)
        return image, filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    # Requirement: Initialize MultiTaskResNet(use_aux=True) (Structure must match the trained model).
    # Since best_model.pth does not have segmentation head weights, we must set use_aux=False
    model = MultiTaskResNet(use_aux=True)
    
    # Load weights
    # We prefer best_model.pth if it exists, otherwise last_model.pth
    ckpt_path = 'best_model.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'last_model.pth'
    
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using random weights.")
    
    model.to(device)
    model.eval()

    # Transforms (Same as training)
    test_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(args.test_data_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    results = {}

    print(f"Starting Inference on {len(test_dataset)} images...")
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            out_cls, _ = model(images)
            probs = torch.sigmoid(out_cls)
            
            # Apply per-class thresholds
            # Index 0 (No Defect): 0.5
            # Index 1 (Chipped):   0.5
            # Index 2 (Scratch):   0.4 (lower threshold to boost recall)
            # Index 3 (Stain):     0.3 (lower threshold to boost recall)
            thresholds = torch.tensor([0.5, 0.5, 0.4, 0.3], device=device)
            preds = (probs > thresholds).int().cpu().numpy()
            
            for i in range(len(filenames)):
                filename = filenames[i]
                pred_row = preds[i] # Shape (4,)
                
                # Convert prediction vector to list of output strings
                label_list = []
                for idx, val in enumerate(pred_row):
                    if val == 1:
                        label_list.append(INDEX_TO_LABEL[idx])
                
                # Ensure consistency: if list matches expectations.
                # If everything is 0, list is empty.
                results[filename] = label_list

    # Save to [Student_ID].json
    output_filename = 'PB23071385.json'
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_filename}")

if __name__ == '__main__':
    main()
