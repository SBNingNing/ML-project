import argparse
import os
import sys
import json
import glob
import re
import numpy as np
import torch
from PIL import Image

# ==========================================
# 必须导入 model 来获取网络定义
# 请确保你的 model.py 是修复了 dW/db 初始化问题的版本！
# ==========================================
from model import Config, build_model

# 你的学号 (用于命名输出的 JSON 文件)
STUDENT_ID = "PB23071397"

def find_latest_model(dir_path):
    """
    自动寻找模型文件
    优先级: best_model.pth > model_final.pth > model_epoch_XX.pth
    """
    candidates = ['best_model.pth', 'best_loss_model.pth','model_final.pth']
    for name in candidates:
        path = os.path.join(dir_path, name)
        if os.path.exists(path):
            return path

    search_pattern = os.path.join(dir_path, 'model_epoch_*.pth')
    files = glob.glob(search_pattern)
    
    if not files:
        return None
        
    def extract_epoch(filename):
        match = re.search(r'model_epoch_(\d+).pth', filename)
        return int(match.group(1)) if match else -1
        
    files.sort(key=extract_epoch, reverse=True)
    return files[0]

def get_inference_loader(data_dir):
    """
    纯推理加载器：只读取图片
    """
    # 兼容路径处理: 确保指向包含图片的目录
    if data_dir.endswith('img'):
        img_dir = data_dir
    else:
        # 如果传入的是 dataset/test，则找 dataset/test/img
        # 如果 dataset/test/img 不存在，则假设图片就在 dataset/test 下
        sub_img_dir = os.path.join(data_dir, 'img')
        if os.path.exists(sub_img_dir):
            img_dir = sub_img_dir
        else:
            img_dir = data_dir
    
    # 获取所有 png 图片
    files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    files.sort() # 排序保证稳定性
    
    batch_size = Config.batch_size
    # CPU 环境下减小 batch size 防止内存压力
    if Config.device == 'cpu':
        batch_size = 16

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        X = []
        valid_files = [] 
        
        for f in batch_files:
            try:
                img_path = os.path.join(img_dir, f)
                
                # 图像预处理 (与训练保持一致)
                img = Image.open(img_path).convert('RGB').resize((Config.img_size, Config.img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1) # HWC -> CHW
                
                X.append(arr)
                valid_files.append(f)
            except: pass
        
        if X:
            x_arr = np.array(X, dtype=np.float32)
            yield torch.from_numpy(x_arr).float().to(Config.device), valid_files

def run_inference(test_dir):
    results = {}
    
    try:
        # 1. 自动定位并加载模型
        model_path = find_latest_model(Config.current_dir)
        if model_path is None:
            raise FileNotFoundError("No model (.pth) found in script directory!")
        
        # print(f"Loading model from {os.path.basename(model_path)}...") 

        model = build_model()
        model.load(model_path)
        
        # 2. 推理循环
        for X, filenames in get_inference_loader(test_dir):
            # 前向传播 (关闭梯度计算可稍微加速，虽在inference模式下不反传也没事，但习惯更好)
            # 由于是手动层，这里只需确保 training=False
            out = model.forward(X, training=False)
            
            # 转为 numpy 数组
            probs = out.cpu().numpy().flatten()
            
            # 根据阈值判定 (0 或 1)
            predictions = (probs > Config.threshold).astype(int)
            
            # 3. 格式化结果并存入字典
            for fname, pred in zip(filenames, predictions):
                # Key: 去掉扩展名 (.png)
                key_name = os.path.splitext(fname)[0]
                
                # Value: 转为布尔值 (True 代表 defective/1, False 代表 non-defective/0)
                # 注意：numpy 的 bool_ 类型 json 不认，必须转为 python 原生 bool
                is_defective = bool(pred == 1)
                
                results[key_name] = is_defective
        
        # 4. 保存为 JSON 文件
        output_filename = f"{STUDENT_ID}.json"
        output_path = os.path.join(Config.current_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Inference done. Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    args = parser.parse_args()
    
    run_inference(args.test_data_path)
