import os
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from model import Config

def get_balanced_train_paths(data_dir):
    """加载 dataset/train 并进行正负样本平衡"""
    train_dir = os.path.join(data_dir, 'train')
    img_dir = os.path.join(train_dir, 'img')
    txt_dir = os.path.join(train_dir, 'txt')
    
    if not os.path.exists(img_dir):
        # 兼容性：如果找不到，尝试自动修复一次路径
        if os.path.exists(Config.data_dir):
            train_dir = os.path.join(Config.data_dir, 'train')
            img_dir = os.path.join(train_dir, 'img')
            txt_dir = os.path.join(train_dir, 'txt')
        
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Train dir not found: {img_dir}")

    all_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    pos, neg = [], []
    for f in all_files:
        path = os.path.join(img_dir, f)
        has_label = os.path.exists(os.path.join(txt_dir, f.replace('.png', '.txt')))
        if has_label: pos.append((path, 1))
        else: neg.append((path, 0))
    
    # 策略：正样本x4，负样本保留1.2倍的正样本量
    pos_multiplier = 4
    balanced_pos = pos * pos_multiplier
    target_neg = int(len(balanced_pos) * 1.2)
    random.shuffle(neg)
    balanced_neg = neg[:target_neg]
    
    data = balanced_pos + balanced_neg
    random.shuffle(data)
    print(f"[Dataset] Train Balanced: Pos(x{pos_multiplier})={len(balanced_pos)}, Neg={len(balanced_neg)}")
    return data

def get_test_paths(data_dir):
    """加载 dataset/test (无平衡)"""
    test_dir = os.path.join(data_dir, 'test')
    img_dir = os.path.join(test_dir, 'img')
    txt_dir = os.path.join(test_dir, 'txt')
    
    data = []
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.endswith('.png'):
                lbl = 1 if os.path.exists(os.path.join(txt_dir, f.replace('.png', '.txt'))) else 0
                data.append((os.path.join(img_dir, f), lbl))
    return data

def data_loader(data_list, batch_size, training=True):
    random.shuffle(data_list)
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        X, Y = [], []
        for path, label in batch:
            try:
                img = Image.open(path).convert('RGB').resize((Config.img_size, Config.img_size))
                if training:
                    # 强增强
                    if random.random() > 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if random.random() > 0.5: img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    rot = random.choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                    if rot: img = img.transpose(rot)
                    if random.random() > 0.3: img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
                    if random.random() > 0.3: img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.4))
                
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)
                X.append(arr)
                Y.append(label)
            except: pass
        if X:
            x_arr = np.array(X, dtype=np.float32)
            y_arr = np.array(Y, dtype=np.float32)
            yield torch.from_numpy(x_arr).float().to(Config.device), \
                  torch.from_numpy(y_arr).float().unsqueeze(1).to(Config.device)
