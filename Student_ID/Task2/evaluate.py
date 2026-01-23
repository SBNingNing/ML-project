import json
import os
import glob

def evaluate():
    # Paths
    PRED_FILE = 'PB23071385.json' 
    GT_DIR = os.path.join('..', '..', 'dataset', 'test', 'txt')
    
    if not os.path.exists(PRED_FILE):
        print(f"Error: {PRED_FILE} not found. Run For_TA_test.py first.")
        return

    with open(PRED_FILE, 'r') as f:
        preds = json.load(f)
        
    # Global counters (Micro calculation)
    tp_all = 0
    fp_all = 0
    fn_all = 0
    
    # Per-class counters (Macro calculation)
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
        # 1. Get Ground Truth Labels
        gt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(GT_DIR, gt_filename)
        
        gt_labels = set()
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            has_defect = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    has_defect = True
                    class_id = int(parts[0]) 
                    
                    # Map YOLO ID to Output String
                    if class_id == 0: gt_labels.add("2")   # Chipped
                    elif class_id == 1: gt_labels.add("3") # Scratch
                    elif class_id == 2: gt_labels.add("4") # Stain
            
            if not has_defect:
                gt_labels.add("1") # No Defect
        else:
            gt_labels.add("1") # No Defect
            
        # 2. Convert Prediction to Set
        pred_set = set(pred_labels)
        
        # 3. Update Global Stats (for Micro)
        intersection = pred_set.intersection(gt_labels)
        tp_all += len(intersection)
        fp_all += len(pred_set - gt_labels)
        fn_all += len(gt_labels - pred_set)
        
        if pred_set == gt_labels:
            correct_count += 1
            
        # 4. Update Per-class Stats (for Macro)
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
        
    # --- Metrics Calculation ---
    epsilon = 1e-15 
    
    # 1. Overall Exact Match Accuracy
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    
    # 2. Overall Micro F1 (Weighted by instance freq implicitly)
    precision_micro = tp_all / (tp_all + fp_all + epsilon)
    recall_micro = tp_all / (tp_all + fn_all + epsilon)
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + epsilon)
    
    # 3. Overall Macro F1 (Average of per-class F1)
    f1_list = []
    all_classes = ["1", "2", "3", "4"]
    class_names = {"1": "No Defect", "2": "Chipped", "3": "Scratch", "4": "Stain"}
    
    print("-" * 40)
    print("Per-Class Performance:")
    for cls in all_classes:
        stats = class_stats[cls]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        p = tp / (tp + fp + epsilon)
        r = tp / (tp + fn + epsilon)
        f1 = 2 * (p * r) / (p + r + epsilon)
        f1_list.append(f1)
        
        print(f"  Class {cls} ({class_names[cls]}): P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

    f1_macro = sum(f1_list) / len(f1_list)
    
    print("-" * 40)
    print(f"Total Samples: {total_samples}")
    print(f"Overall Exact Match Accuracy: {accuracy:.4f}")
    print(f"Overall Micro F1-score:       {f1_micro:.4f}")
    print(f"Overall Macro F1-score:       {f1_macro:.4f}")
    print("-" * 40)
    
    # 4. Defect-Only Metrics (Optional but useful for analysis)
    # Re-calculate micro F1 only for classes 2, 3, 4
    tp_def = sum([class_stats[c]["tp"] for c in ["2", "3", "4"]])
    fp_def = sum([class_stats[c]["fp"] for c in ["2", "3", "4"]])
    fn_def = sum([class_stats[c]["fn"] for c in ["2", "3", "4"]])
    
    prec_def = tp_def / (tp_def + fp_def + epsilon)
    rec_def = tp_def / (tp_def + fn_def + epsilon)
    f1_def = 2 * (prec_def * rec_def) / (prec_def + rec_def + epsilon)
    
    print(f"Defect-Only Micro F1-score:   {f1_def:.4f}")
    print("-" * 40)

if __name__ == '__main__':
    evaluate()