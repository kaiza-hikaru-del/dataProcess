import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置禁用 GUI 交互
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import concurrent.futures
from typing import List, Tuple
import itertools  # 新增 itertools 用于笛卡尔积
from tqdm.rich import tqdm

# ======================
#  Step 1: 数据重组与裁剪
# ======================

def process_sample(sample_dir: Path) -> None:  # 修改参数类型为 Path
    """处理单个sample目录"""    
    # 遍历所有原始图像
    img_files = list(sample_dir.glob("*.jpg"))
    for img_file in tqdm(img_files):
        # 解析文件名
        parts = img_file.stem.split("-")
        if len(parts) != 2:
            continue
        field_no, order = parts
        
        # 创建目标目录结构
        field_dir = sample_dir / f"field{field_no}"
        
        # 读取并处理图像
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        # 中心裁剪 (2448x2048 -> 2240x2016)
        cropped = img[16:16+2016, 104:104+2240]
        
        # 使用笛卡尔积简化循环
        roi_height, roi_width = 224, 224
        for i, j in itertools.product(range(9), range(10)):  # 9x10 笛卡尔积
            roi_no = i * 10 + j
            roi = cropped[i*roi_height:(i+1)*roi_height,
                          j*roi_width:(j+1)*roi_width]
            
            # 直接使用 pathlib 创建目录
            roi_dir = field_dir / f"roi{roi_no:03d}"
            roi_dir.mkdir(parents=True, exist_ok=True)  # 替换 create_dir
            cv2.imwrite(str(roi_dir / f"{order}.jpg"), roi)

        # 删除原始图像
        img_file.unlink()

# ======================
#  Step 2: 清晰度计算与重标注
# ======================

def sobel_sharpness(image: np.ndarray) -> float:
    """基于Sobel算子的清晰度计算"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.medianBlur(gray, ksize=7)
    sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

def dof_calculate(l: float, n: float, NA: float, m: float, e: float) -> float:
    """景深计算"""
    return (l * n) / (NA**2) + (n * e) / (m * NA)

# 步长映射表
STEP_MAPPING = {
    (10, 0.3): -1600,
    (20, 0.7): -300,
    (40, 0.65): -300,
    (100, 0.8): -200
}

def process_field(field_dir: Path, step_size: float = 0) -> None:  # 修改参数类型为 Path
    """处理单个field目录"""
    # 解析参数
    category_part = field_dir.parent.parent.name
    mag_str, na_str, _ = category_part.split("_", 2)
    magnification = float(mag_str[:-1])
    na = float(na_str)
    
    # 自动确定步长逻辑
    if step_size == 0:
        step_size = STEP_MAPPING.get((magnification, na), 1000)
    
    # 遍历所有ROI目录
    roi_dirs = list(field_dir.glob("roi*"))
    for roi_dir in tqdm(roi_dirs):
        # 收集所有图像并排序
        img_files = sorted(roi_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        orders = [int(f.stem) for f in img_files]
        
        # 计算清晰度曲线
        sharpness_values = []
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            img = cv2.resize(img, (256, 256))
            sharpness_values.append(sobel_sharpness(img))
        
        # 找到最佳对焦位置
        max_idx = np.argmax(sharpness_values)
        max_order = orders[max_idx]
        
        # 生成CSV数据
        data = {
            "image_path": [str(f.relative_to(field_dir.parent.parent.parent)) for f in img_files],
            "sobel_sharpness": sharpness_values,
            "defocus_label": [(o - max_order) * step_size for o in orders],
            "magnification": magnification,
            "NA": na,
        }
        
        # 计算景深相关参数
        dof = dof_calculate(550, 1.0, na, magnification, 3450)
        data["defocus_dof_label"] = [l/dof for l in data["defocus_label"]]
        
        # 保存CSV文件
        csv_filename = f"{category_part}-{field_dir.parent.name}-{field_dir.name}-{roi_dir.name}.csv"
        csv_path = field_dir.parent.parent.parent / ".info" / csv_filename
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        # 生成并保存曲线图
        plt.figure(figsize=(7, 7))
        plt.plot(orders, sharpness_values)
        plt.scatter(max_idx, sharpness_values[max_idx], color='red', s=50, zorder=5, 
                    label=f'Max Sharpness ({max_idx}, {sharpness_values[max_idx]:.2f})')
        plt.xlabel('Z-Stack Order')
        plt.ylabel('Sharpness Value')
        plt.title('Sharpness Variation through Z-Stack')
        plt.grid(True, linestyle='--', alpha=0.7, which='both')
        plt.legend(loc='best', frameon=True)
        plt.tight_layout()
        plt.savefig(field_dir.parent.parent.parent / ".curve" / csv_filename.replace(".csv", ".jpg"), dpi=96)
        plt.close()
        
        # 保存最佳对焦图像
        best_img = cv2.imread(str(img_files[max_idx]))
        cv2.imwrite(str(field_dir.parent.parent.parent / ".infocus" / csv_filename.replace(".csv", ".jpg")), best_img)

# ======================
#  Step 3: 数据集划分
# ======================

def process_category(category_dir: Path, ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> None:
    """处理单个类别目录"""
    print(f"process_category: {category_dir}")
    info_dir = category_dir.parent / ".info"
    
    # ========== 新增排除逻辑 ==========
    # 构造排除文件路径
    exclude_dir = category_dir.parent / ".exclude"
    exclude_file = exclude_dir / f"{category_dir.name}.csv"
    
    # 初始化排除集合
    exclude_set = set()
    
    if exclude_file.exists():
        try:
            # 读取排除CSV
            exclude_df = pd.read_csv(exclude_file, dtype=str)
            
            # 标准化编号格式为三位数
            exclude_df["sample_no"] = exclude_df["sample_no"].str.zfill(3)
            exclude_df["field_no"] = exclude_df["field_no"].str.zfill(3)
            exclude_df["roi_no"] = exclude_df["roi_no"].str.zfill(3)
            
            # 生成匹配模式
            for _, row in exclude_df.iterrows():
                pattern = (
                    f"{category_dir.name}-"
                    f"sample{row['sample_no']}-"
                    f"field{row['field_no']}-"
                    f"roi{row['roi_no']}.csv"
                )
                exclude_set.add(pattern)
        except Exception as e:
            print(f"Error reading exclude file {exclude_file}: {e}")
    # ========== 排除逻辑结束 ==========
    
    # 收集所有CSV文件并过滤
    csv_files = [f for f in info_dir.glob(f"{category_dir.name}-*.csv") 
                 if f.name not in exclude_set]
    
    # 划分数据集
    np.random.shuffle(csv_files)
    split_points = [int(len(csv_files)*ratios[0]), 
                    int(len(csv_files)*(ratios[0]+ratios[1]))]
    splits = np.split(csv_files, split_points)
    
    # 保存划分结果
    for split_name, files in zip([".train", ".val", ".test"], splits):
        split_dir = category_dir.parent / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, split_dir / f.name)

# ======================
#  Main Execution
# ======================

def main(dataset_root: Path, step: int = 0) -> None:
    """主执行函数"""
    # Step 1: 并行处理所有sample
    if step == 1 or step == 0:
        samples = list(dataset_root.glob("*/sample*"))  # 直接使用 Path
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(process_sample, samples)
    
    # Step 2: 处理更深层级的 field
    if step == 2 or step == 0:
        fields = list(dataset_root.glob("*/sample*/field*"))  # 修正路径层级
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(process_field, fields)
    
    # Step 3: 处理所有类别
    if step == 3 or step == 0:
        categories = list(dataset_root.glob("[!.]*"))
        for cat_dir in categories:
            process_category(cat_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                       help="数据集根目录路径")
    parser.add_argument("--step", type=int, choices=[0,1,2,3], default=0,
                       help="执行步骤 (0=全部,1=预处理,2=重标注,3=数据集划分)")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    # 创建必要目录
    [ (dataset_path / subdir).mkdir(exist_ok=True) 
      for subdir in (".info", ".curve", ".infocus", ".train", ".val", ".test", ".exclude") ]
    
    main(dataset_path, args.step)