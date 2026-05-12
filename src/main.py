# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import sys
sys.path.append(r'D:\recommend\BM3-master\src')
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'
from sklearn.manifold import TSNE
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BM3', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--use_gpu', type=int, default=1, help='whether to use GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='which GPU to use')

    args, _ = parser.parse_known_args()

    config_dict = {
        'use_gpu': bool(args.use_gpu),
        'gpu_id': args.gpu_id
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


def plot_tsne(model, test_loader, device, save_path="tsne_disentanglement.png"):
    """
    专门用于生成 t-SNE 解耦可视化图的函数 (4 个特征版本，纯圆点无形状区分)
    """
    print("\n--- Starting t-SNE Feature Extraction ---")
    model.eval()
    
    feats = {'c_l': [], 's_l': [], 's_v': []}
    
    with torch.no_grad():
        for data in test_loader:
            incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))
            out = model(incomplete_input)
            
            # 安全检查：确保模型确实输出了这些特征
            if 'c_l' not in out:
                print("Error: Features like 'c_l' not found in model output! Please update DGMoE.py forward() return dict.")
                return
                
            for key in feats.keys():
                f = out[key]
                # 如果特征是 (Batch, Seq_len, Dim)，在序列维度求平均以匹配 t-SNE 的要求
                if f.dim() == 3:
                    f = f.mean(dim=1) 
                feats[key].append(f.cpu().numpy())

    print("Concatenating features...")
    for key in feats.keys():
        feats[key] = np.concatenate(feats[key], axis=0)

    # 将 4 种特征垂直堆叠，喂给 t-SNE
    all_features = np.vstack([feats['c_l'], 
                              feats['s_l'], feats['s_a'], feats['s_v']])
    
    # 生成对应的颜色标签
    num_samples = len(feats['c_l'])
    labels = (
        ['Shared'] * num_samples +
        ['Specific-Text'] * num_samples + 
        ['Specific-Vision'] * num_samples
    )
    
    print(f"Total feature shape: {all_features.shape}. Running t-SNE (this might take a minute or two)...")
    # 使用更高质量的 t-SNE 参数
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, init='pca')
    features_2d = tsne.fit_transform(all_features)
    
    print("Plotting and saving...")
    plt.figure(figsize=(10, 8))
    
    # 定义色板：Shared 为高级的灰色，Specific 为经典三原色
    palette = {
        'Shared': '#7f7f7f',        # 中性深灰
        'Specific-Text': '#1f77b4', # 蓝
        'Specific-Vision': '#2ca02c'# 绿
    }
    
    # 使用 seaborn 画图，去掉了 style 和 markers 参数，所有点默认为圆形
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=labels,          # 仅控制颜色
        palette=palette,
        alpha=0.75,          # 适当透明度透出底层
        s=60,                # 放大点
        edgecolor='white',   # 加入白色精细描边
        linewidth=0.5
    )
    
    plt.title("t-SNE Visualization of Feature Disentanglement", fontsize=14)
    # 将图例放在图的右侧外面，去掉边框使其更简洁
    # 修改后的代码（放在框内右上角，并加上半透明背景防遮挡）：
    plt.legend(title="Feature Type", loc='upper right', frameon=True, framealpha=0.8, edgecolor='gray')
    plt.tight_layout()
    
    # 保存为高清 PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot successfully saved to: {save_path}\n")