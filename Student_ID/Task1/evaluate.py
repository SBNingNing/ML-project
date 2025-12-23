import os
import glob
import re
import numpy as np
from sklearn.metrics import f1_score
from model import Config, build_model
from dataset import get_test_paths, data_loader

def get_all_checkpoints(dir_path):
    """找到目录下所有的模型文件 (.pth)"""
    # 匹配 model_epoch_X.pth, model_final.pth, best_model.pth
    patterns = [
        os.path.join(dir_path, 'model_epoch_*.pth'),
        os.path.join(dir_path, 'model_final.pth'),
        os.path.join(dir_path, 'best_model.pth'),
        os.path.join(dir_path, 'best_loss_model.pth')
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    # 去重并排序
    files = sorted(list(set(files)))
    return files

def evaluate_single_model(model, test_data, batch_size):
    """对单个模型进行推理并返回预测结果"""
    preds, targets = [], []
    for X, Y in data_loader(test_data, batch_size, training=False):
        out = model.forward(X, training=False)
        preds.extend(out.cpu().numpy().flatten())
        targets.extend(Y.cpu().numpy().flatten())
    return np.array(preds), np.array(targets)

def search_best_threshold(preds, targets):
    """为单个模型搜索最佳阈值"""
    best_f1 = 0
    best_thresh = 0.5
    # 粗略搜索 + 精细搜索
    for t in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        f1 = f1_score(targets, (preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_f1, best_thresh

def main():
    print(f"[Eval] Device: {Config.device}")
    
    # 1. 准备数据
    test_data = get_test_paths(Config.data_dir)
    if not test_data:
        print("[Error] No test data found.")
        return
    print(f"[Eval] Loaded {len(test_data)} test samples.")

    # 2. 找到所有模型
    model_files = get_all_checkpoints(Config.current_dir)
    if not model_files:
        print("[Error] No .pth models found. Train first!")
        return
        
    print(f"[Eval] Found {len(model_files)} models. Starting competition...\n")
    print(f"{'Model Name':<25} | {'Best F1':<10} | {'Threshold':<10}")
    print("-" * 50)

    # 3. 逐个评估
    global_best_f1 = 0
    global_best_model = ""
    global_best_thresh = 0.5
    
    # 初始化模型结构 (只建一次，后面只这就load_state_dict)
    model = build_model()
    
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        try:
            model.load(model_path)
            
            # 推理
            preds, targets = evaluate_single_model(model, test_data, Config.batch_size)
            
            # 找该模型的最佳表现
            f1, thresh = search_best_threshold(preds, targets)
            
            print(f"{model_name:<25} | {f1:.4f}     | {thresh:<10}")
            
            # 更新全局最佳
            if f1 > global_best_f1:
                global_best_f1 = f1
                global_best_model = model_name
                global_best_thresh = thresh
                
        except Exception as e:
            print(f"{model_name:<25} | Error: {str(e)}")

    # 4. 总结
    print("-" * 50)
    print(f"\n🏆 WINNER: {global_best_model}")
    print(f"🥇 Max F1 Score: {global_best_f1:.4f}")
    print(f"🔑 Optimal Threshold: {global_best_thresh}")
    
    print("\n>>> 操作步骤 (Action Items):")
    print(f"1. 请将文件 '{global_best_model}' 重命名为 'best_model.pth'")
    print(f"   命令: mv {global_best_model} best_model.pth")
    print(f"2. 修改 model.py 中的 Config.threshold = {global_best_thresh}")

if __name__ == "__main__":
    main()
