import json
import os
import glob

def evaluate():
    # Paths
    PRED_FILE = 'pred_result.json'
    GT_DIR = os.path.join('..', '..', 'dataset', 'test', 'txt')
    IMG_DIR = os.path.join('..', '..', 'dataset', 'test', 'img')
    
    if not os.path.exists(PRED_FILE):
        print(f"Error: {PRED_FILE} not found. Run For_TA_test.py first.")
        return

    with open(PRED_FILE, 'r') as f:
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
        # pred_labels is a list like ["1", "2"]
        
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
    evaluate()
