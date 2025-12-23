import torch
import torch.nn as nn
import cv2
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from model_mtl import MultiTaskResNet

def test():
    # Hyperparameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    THRESHOLD = 0.5
    
    # Paths
    # Assuming script is in Student_ID/Task2/
    # Test images in ../../dataset/test/img
    TEST_IMG_DIR = os.path.join('..', '..', 'dataset', 'test', 'img')
    MODEL_PATH = 'best_model.pth'
    OUTPUT_FILE = 'pred_result.json'
    
    if not os.path.exists(TEST_IMG_DIR):
        print(f"Warning: Test image directory {TEST_IMG_DIR} does not exist.")
        return

    # Load Model
    model = MultiTaskResNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Error: Model file {MODEL_PATH} not found. Please train first.")
        return
        
    model.eval()
    
    # Get test images
    img_paths = glob.glob(os.path.join(TEST_IMG_DIR, '*.png'))
    img_paths.sort()
    
    results = {}
    
    print(f"Found {len(img_paths)} test images. Starting inference...")
    
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Inference"):
            filename = os.path.basename(img_path)
            file_key = os.path.splitext(filename)[0]
            
            # Preprocess
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (320, 320))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) # [1, 3, 320, 320]
            image = image.to(DEVICE)
            
            # Inference
            out_cls, _ = model(image) # Discard segmentation output
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(out_cls)
            preds = (probs > THRESHOLD).cpu().numpy()[0] # [4,] boolean array
            
            # Map to output format
            # Internal Index 0 (No Defect) -> Output "1"
            # Internal Index 1 (Chipped)   -> Output "2"
            # Internal Index 2 (Scratch)   -> Output "3"
            # Internal Index 3 (Stain)     -> Output "4"
            
            pred_classes = []
            if preds[0]: pred_classes.append("1")
            if preds[1]: pred_classes.append("2")
            if preds[2]: pred_classes.append("3")
            if preds[3]: pred_classes.append("4")
            
            results[file_key] = pred_classes
            
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Inference complete. Results saved to {OUTPUT_FILE}")
    
    # Evaluate
    evaluate_results(OUTPUT_FILE)

def evaluate_results(pred_file):
    GT_DIR = os.path.join('..', '..', 'dataset', 'test', 'txt')
    
    if not os.path.exists(pred_file):
        print(f"Error: {pred_file} not found.")
        return

    with open(pred_file, 'r') as f:
        preds = json.load(f)
        
    # Global counters (All classes)
    tp_all = 0
    fp_all = 0
    fn_all = 0
    
    # Per-class counters
    class_stats = {
        "1": {"tp": 0, "fp": 0, "fn": 0}, # No Defect
        "2": {"tp": 0, "fp": 0, "fn": 0}, # Chipped
        "3": {"tp": 0, "fp": 0, "fn": 0}, # Scratch
        "4": {"tp": 0, "fp": 0, "fn": 0}  # Stain
    }
    
    correct_count = 0
    total_samples = 0
    
    print(f"Evaluating {len(preds)} samples...")
    
    for filename, pred_labels in preds.items():
        # Construct GT path
        txt_path = os.path.join(GT_DIR, filename + '.txt')
        
        gt_labels = set()
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            has_defect = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    has_defect = True
                    class_id = int(parts[0]) # 0, 1, 2
                    
                    # Map to Output String
                    # 0 (Chipped) -> "2"
                    # 1 (Scratch) -> "3"
                    # 2 (Stain)   -> "4"
                    
                    if class_id == 0: gt_labels.add("2")
                    elif class_id == 1: gt_labels.add("3")
                    elif class_id == 2: gt_labels.add("4")
            
            if not has_defect:
                gt_labels.add("1") # No Defect
        else:
            # No txt file -> No Defect
            gt_labels.add("1")
            
        # Convert pred to set
        pred_set = set(pred_labels)
        
        # Global stats
        intersection = pred_set.intersection(gt_labels)
        tp_all += len(intersection)
        fp_all += len(pred_set - gt_labels)
        fn_all += len(gt_labels - pred_set)
        
        if pred_set == gt_labels:
            correct_count += 1
            
        # Per-class stats
        for cls in ["1", "2", "3", "4"]:
            is_in_pred = cls in pred_set
            is_in_gt = cls in gt_labels
            
            if is_in_pred and is_in_gt:
                class_stats[cls]["tp"] += 1
            elif is_in_pred and not is_in_gt:
                class_stats[cls]["fp"] += 1
            elif not is_in_pred and is_in_gt:
                class_stats[cls]["fn"] += 1
        
        total_samples += 1
        
    # Calculate Metrics
    epsilon = 1e-15 # To avoid division by zero
    
    # 1. Overall Exact Match Accuracy
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    
    # 2. Overall Micro Metrics (All classes)
    precision_all = tp_all / (tp_all + fp_all + epsilon)
    recall_all = tp_all / (tp_all + fn_all + epsilon)
    f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all + epsilon)
    
    print(f"Total Samples: {total_samples}")
    print("-" * 30)
    print(f"Overall Exact Match Accuracy: {accuracy:.4f}")
    print(f"Overall Micro F1-score:       {f1_all:.4f}")
    print("-" * 30)
    
    # 3. Defect Classes Metrics (2, 3, 4)
    defect_classes = ["2", "3", "4"]
    class_names = {"2": "Chipped", "3": "Scratch", "4": "Stain"}
    
    tp_def_micro = 0
    fp_def_micro = 0
    fn_def_micro = 0
    
    print("Defect Classes Evaluation:")
    for cls in defect_classes:
        stats = class_stats[cls]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        tp_def_micro += tp
        fp_def_micro += fp
        fn_def_micro += fn
        
        p = tp / (tp + fp + epsilon)
        r = tp / (tp + fn + epsilon)
        f1 = 2 * (p * r) / (p + r + epsilon)
        
        print(f"  Class {cls} ({class_names[cls]}): Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")
        
    # 4. Defect Micro Metrics
    prec_def_micro = tp_def_micro / (tp_def_micro + fp_def_micro + epsilon)
    rec_def_micro = tp_def_micro / (tp_def_micro + fn_def_micro + epsilon)
    f1_def_micro = 2 * (prec_def_micro * rec_def_micro) / (prec_def_micro + rec_def_micro + epsilon)
    
    print("-" * 30)
    print(f"Defect-Only Micro Precision: {prec_def_micro:.4f}")
    print(f"Defect-Only Micro Recall:    {rec_def_micro:.4f}")
    print(f"Defect-Only Micro F1-score:  {f1_def_micro:.4f}")

if __name__ == '__main__':
    test()
