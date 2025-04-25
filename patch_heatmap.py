from PIL import Image
import numpy as np
import torch
from models import get_model
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def slice_image(img, window_size=224):
    # 如果输入是路径字符串，则打开图像
    if isinstance(img, str):
        img = Image.open(img)

    # 储存一个centercrop的版本
    center_cropped = transforms.CenterCrop(224)(img)
    
    # 获取图像尺寸
    width, height = img.size
    
    # 计算完整的行数和列数
    cols = width // window_size
    rows = height // window_size
    
    # 计算是否需要额外的右侧和底部窗口
    extra_right = width % window_size > 0
    extra_bottom = height % window_size > 0
    
    # 如果有额外的右侧列，总列数加1
    total_cols = cols + (1 if extra_right else 0)
    # 如果有额外的底部行，总行数加1
    total_rows = rows + (1 if extra_bottom else 0)
    
    # 创建一个二维列表用于存储裁剪的窗口
    windows_grid = [[None for _ in range(total_cols)] for _ in range(total_rows)]
    
    # 处理完整的非重叠窗口
    for r in range(rows):
        for c in range(cols):
            left = c * window_size
            upper = r * window_size
            right = left + window_size
            lower = upper + window_size
            
            window = img.crop((left, upper, right, lower))
            windows_grid[r][c] = window
    
    # 处理最右侧的窗口（如果有）
    if extra_right:
        for r in range(rows):
            left = width - window_size
            upper = r * window_size
            right = width
            lower = upper + window_size
            
            window = img.crop((left, upper, right, lower))
            windows_grid[r][cols] = window
    
    # 处理底部的窗口（如果有）
    if extra_bottom:
        for c in range(cols):
            left = c * window_size
            upper = height - window_size
            right = left + window_size
            lower = height
            
            window = img.crop((left, upper, right, lower))
            windows_grid[rows][c] = window
    
    # 处理右下角的窗口（如果同时有额外的行和列）
    if extra_right and extra_bottom:
        left = width - window_size
        upper = height - window_size
        right = width
        lower = height
        
        window = img.crop((left, upper, right, lower))
        windows_grid[rows][cols] = window
    
    return center_cropped, windows_grid

def calculate_heat_map(ckpt, gpu_id, center_cropped, windows_grid):
    model = get_model("DINOv2-LoRA:dinov2_vitl14")
    state_dict = torch.load(ckpt, map_location='cpu')['model']
    #model.fc.load_state_dict(state_dict)
    model.load_state_dict(state_dict)

    model.cuda(gpu_id)
    model.eval()

    center_cropped_tensor = transforms.ToTensor()(center_cropped).unsqueeze(0).cuda(gpu_id)
    original_score = torch.sigmoid(model(center_cropped_tensor)).item()

    scores = []
    for i in range(len(windows_grid)):
        scores.append([])
        for j in range(len(windows_grid[i])):
            patch = windows_grid[i][j]
            patch_tensor = transforms.ToTensor()(patch).unsqueeze(0).cuda(gpu_id)
            score = torch.sigmoid(model(patch_tensor)).item()
            scores[i].append(score)

    return original_score, scores


def plot_heatmap(data_grid, save_file, title="Heat Map", cmap="viridis", annot=True, figsize=(10, 8)):
    """
    绘制二维数组的热力图
    
    参数:
    - data_grid: 二维数组或列表的列表
    - title: 图表标题
    - cmap: 颜色映射，例如 'viridis', 'hot', 'coolwarm', 'YlGnBu', 'RdBu'
    - annot: 是否在每个单元格显示数值
    - figsize: 图表大小 (宽, 高)
    """
    # 将输入转换为numpy数组
    data_array = np.array(data_grid)
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 使用seaborn绘制热力图
    ax = sns.heatmap(data_array, 
                     annot=annot,  # 显示单元格的值
                     fmt=".2f",    # 数值格式化（保留两位小数）
                     cmap=cmap,    # 颜色映射
                     linewidths=.5,  # 单元格边框宽度
                     cbar=True)     # 显示颜色条
    
    # 设置标题和轴标签
    plt.title(title)
    plt.xlabel('row')
    plt.ylabel('col')
    
    # 如果您希望行和列的索引从1开始而不是0
    # plt.yticks(np.arange(data_array.shape[0]), np.arange(1, data_array.shape[0] + 1))
    # plt.xticks(np.arange(data_array.shape[1]), np.arange(1, data_array.shape[1] + 1))
    
    # 显示图表
    plt.tight_layout()
    plt.savefig(save_file)
    
    return ax

def plot_overlapped_heatmap_with_precise_windows(original_image, scores, window_size=224, save_file="overlap_heatmap.png", 
                                               title="Heat Map Overlay", cmap="viridis", alpha=0.7, figsize=(12, 10)):
    """
    将热力图与原图重叠绘制，精确考虑重叠窗口的实际位置
    
    参数:
    - original_image: 原始PIL图像
    - scores: 得分二维数组
    - window_size: 裁剪窗口大小
    - save_file: 保存文件路径
    - title: 图表标题
    - cmap: 颜色映射
    - alpha: 热力图透明度
    - figsize: 图表大小
    """
    # 将输入的分数转换为numpy数组
    scores_array = np.array(scores)
    
    # 获取原图尺寸
    width, height = original_image.size
    
    # 获取热力图的行数和列数
    heat_rows, heat_cols = scores_array.shape
    
    # 计算实际的窗口位置
    window_positions = []
    
    # 计算完整的行数和列数（不包括可能的额外行列）
    full_cols = width // window_size
    full_rows = height // window_size
    
    # 构建窗口位置列表
    for r in range(heat_rows):
        row_positions = []
        for c in range(heat_cols):
            # 默认位置（无重叠的情况）
            left = c * window_size
            upper = r * window_size
            
            # 如果是最右侧的列且不是完整的列
            if c == heat_cols - 1 and c >= full_cols:
                left = width - window_size
                
            # 如果是最底部的行且不是完整的行
            if r == heat_rows - 1 and r >= full_rows:
                upper = height - window_size
                
            # 窗口的右边和底边
            right = left + window_size
            lower = upper + window_size
            
            # 添加窗口位置
            row_positions.append((left, upper, right, lower))
        window_positions.append(row_positions)
    
    # 获取分数的最小值和最大值
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    if min_score == max_score:
        normalized_scores = np.zeros_like(scores_array)
    else:
        normalized_scores = (scores_array - min_score) / (max_score - min_score)
    
    # 获取颜色映射
    cmap_func = plt.cm.get_cmap(cmap)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 显示原始图像
    ax.imshow(np.array(original_image))
    
    # 绘制每个窗口的热力图和边界
    for r in range(heat_rows):
        for c in range(heat_cols):
            left, upper, right, lower = window_positions[r][c]
            score = scores_array[r][c]
            norm_score = normalized_scores[r][c]
            
            # 获取对应的颜色
            color = cmap_func(norm_score)
            
            # 创建一个半透明的矩形
            rect = plt.Rectangle((left, upper), right-left, lower-upper, 
                                fill=True, alpha=alpha, color=color, 
                                edgecolor='white', linestyle='--', linewidth=1)
            ax.add_patch(rect)
            
            # 添加分数文本
            ax.text((left+right)/2, (upper+lower)/2, f"{score:.2f}", 
                     ha='center', va='center', color='white', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, pad=0.3))
    
    # 创建一个标量映射对象并添加颜色条 - 修复了之前的问题
    norm = plt.Normalize(min_score, max_score)
    sm = plt.cm.ScalarMappable(cmap=cmap_func, norm=norm)
    sm.set_array([])  # 必须设置一个数组，即使是空的
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Score')
    
    # 设置标题和轴标签
    ax.set_title(title)
    ax.set_xlabel(f'Width: {width}px')
    ax.set_ylabel(f'Height: {height}px')
    
    # 隐藏坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 显示图表
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    
    return save_file


# 使用示例
if __name__ == "__main__":
    # 替换为您的图像路径
    image_path = "/root/autodl-tmp/AIGC_data/Chameleon/test/0_real/0010cc9d-66ff-460a-bf13-1e0c3b6ab603.jpg"
    ckpt = "/root/autodl-tmp/code/VAE_RESIZE_AIGC_detection/checkpoints/xl_mse_ema/model_iters_25000.pth"

    # 获取文件名（包含扩展名）
    img_name_with_extension = image_path.split("/")[-1]
    # 获取文件名（不含扩展名）
    img_name_without_extension = img_name_with_extension.split(".")[0]
    # 类别
    label = "real"

    save_file = f"./{img_name_without_extension}_{label}_heat_map.png"
    gpu_id = 1

    print("\nPatching Image ...")
    center_cropped, windows_grid = slice_image(image_path)

    print("\nCalculating Heat Map ...")
    original_score, scores = calculate_heat_map(ckpt, gpu_id, center_cropped, windows_grid)

    original_image = Image.open(image_path).convert("RGB")

    print("\nPloting Heat Map ...")
    # plot_heatmap(scores, save_file, title=f"Heat Map - {label} - Original Score {original_score}")
    plot_overlapped_heatmap_with_precise_windows(original_image, scores, window_size=224, save_file=save_file, title=f"Heat Map - {label} - Original Score {original_score}", alpha=0.4)