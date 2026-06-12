import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

from train import find_all_wafer_paths
from src.multiprocessing_wrapper import batch_process_by_path
from src.csv_utils import (
	load_matrix_csv,
	matrix_to_sparse_points,
	)
from test_scripts.wafer_plot import create_wafer_mask, draw_die_grid, perform_gpr


def generate_and_create_mirrored_paths(data_in: str, data_out: str) -> List[Dict[str, str]]:
    """
    Scan data_in for wafer directories, check and create mirrored paths under data_out 
    (with a creation message), and return a list of paired input/output path dictionaries.
    
    扫描 data_in 下的晶圆目录，检查并在 data_out 下按需创建对应的镜像路径（附带创建提示），
    同时收集并返回包含输入和输出路径对的任务字典列表。
    
    Args:
        data_in (str): Root directory of the input source dataset (e.g., "data_in/").
                       输入的数据集源根目录（例如 "data_in/"）。
        data_out (str): Root directory for the desired output (e.g., "data_out/").
                        期望输出的目标根目录（例如 "data_out/"）。
                        
    Returns:
        List[Dict[str, str]]: A list of paired paths for jobs, even if directories already existed.
                              成对的绝对路径任务列表（无论文件夹是新创建的还是原本就存在的都会返回）。
                              Example / 示例: [{'input_path': '...', 'output_path': '...'}, ...]
    """
    # Convert string paths to pathlib.Path objects for robust cross-platform path manipulation.
    # 将字符串路径转换为 pathlib.Path 对象，以实现健壮的跨平台路径操作。
    in_root = Path(data_in)
    out_root = Path(data_out)
    
    # Ensure the source input directory actually exists before proceeding.
    # 在继续执行之前，确保输入的源目录真实存在。
    if not in_root.exists():
        raise FileNotFoundError(f"Source directory does not exist / 源目录不存在: {data_in}")
        
    # Define the structural wildcard pattern to match the target wafer directories.
    # 定义匹配晶圆文件夹层级的通配符结构规则。
    search_pattern = "lot_*/csv/wafer_*"
    
    jobs_paths = []
    
    # Traverse all directories matching the pattern. sorted() ensures deterministic order.
    # 遍历所有符合规则的目录。sorted() 保证在不同系统下的遍历顺序完全一致（保证实验复现性）。
    for in_wafer_path in sorted(in_root.glob(search_pattern)):
        if in_wafer_path.is_dir():
            
            # Step 1: Calculate the relative sub-path from data_in (e.g., 'lot_001/csv/wafer_01').
            # 步骤 1：计算出当前 wafer 目录相对于输入根目录的后半段相对路径。
            relative_path = in_wafer_path.relative_to(in_root)
            
            # Step 2: Combine the output root with the relative sub-path to form the target path.
            # 步骤 2：将输出根目录与该相对路径拼接，组合出目标输出的完整路径。
            out_wafer_path = out_root / relative_path
            
            # Step 3: Check existence and create conditionally.
            # 步骤 3：显式判断该镜像路径是否已经存在，从而决定是否创建和打印。
            if not out_wafer_path.exists():
                # Recursively create the target directory and all missing parent folders.
                # 递归创建目标目录以及所有缺失的中间父文件夹（等同于 Linux 的 `mkdir -p`）。
                out_wafer_path.mkdir(parents=True, exist_ok=True)
                
                # Output a real-time message indicating the new directory creation.
                # 📢 只有当文件夹原本不存在、真正执行新建时，才会打印此条消息，避免刷屏。
                print(f"[Create] New directory built: {out_wafer_path}")
            
            # Step 4: Always append the path pair to the job list (whether it was just created or already existed).
            # 步骤 4：无论文件夹是刚刚新建的，还是之前就存在的，都必须把这对绝对路径安全地追加到任务列表中。
            jobs_paths.append((in_wafer_path.resolve(), out_wafer_path.resolve()))
            
    return jobs_paths

def build_test_job_and_label(data_in_path, data_out_path):
    job_list = []
    labels = []
    for (in_path, out_path) in generate_and_create_mirrored_paths(data_in_path, data_out_path):
        job_list.append(
                {
                    "vl_path": in_path / 'Space.csv',
                    "ll_path": in_path / 'MS.csv',
                    "mode": "both",
                    "output_ttf_path": out_path / 'ttf.csv',
                    "output_binary_path": out_path / 'binary_label.csv'
                }
            )
        class_val_matrix = load_matrix_csv(in_path / 'ExistenceClass.csv').astype(np.int32)
        _, _, class_val, _ = matrix_to_sparse_points(class_val_matrix)
        labels.append(np.where(class_val == 3, 1, 0))		# 1 or 2: bad chip (0); 3: good chip (1)

    return job_list, labels

def plot_five_trends(
    x_die, 
    y_die, 
    vl_val, 
    ll_val, 
    t_scores, 
    y_true,          
    y_pred,          
    title='Wafer Spatial Trends, Lifetime & Binary Classification',
    cmap='turbo',
    output_path=None,
    use_log_scale_for_t=True
):
    """
    绘制晶圆综合分析图 (2行3列布局)：
    - [0, 0] / [0, 1]：Space 和 MS 的连续 GPR 空间趋势插值图
    - [0, 2]：预测寿命 t 的矢量方格连续图
    - [1, 0]：真实好/坏芯片分类图 (Per-Die 矢量方格) -> 0画冷色, 1画暖色
    - [1, 1]：模型预测好/坏分类图 (Per-Die 矢量方格) -> 0画冷色, 1画暖色
    - [1, 2]：留空隐藏 (保持版面整洁)
    """
    # --------------------------------------------------------
    # 1. 工艺尺寸 vl 和 ll 的连续空间 GPR 插值
    # --------------------------------------------------------
    print("Performing GPR for Space (vl)...")
    xx, yy, zz_vl, _ = perform_gpr(x_die, y_die, vl_val)
    
    print("Performing GPR for MS (ll)...")
    _, _, zz_ll, _ = perform_gpr(x_die, y_die, ll_val)

    # --------------------------------------------------------
    # 2. 晶圆掩膜 (仅用于 GPR 连续插值图)
    # --------------------------------------------------------
    valid_distance = np.sqrt(x_die**2 + y_die**2)
    wafer_radius = valid_distance.max()

    wafer_mask = create_wafer_mask(xx + 0.5, yy + 0.5, wafer_radius)
    zz_vl = np.where(wafer_mask, zz_vl, np.nan)
    zz_ll = np.where(wafer_mask, zz_ll, np.nan)

    # --------------------------------------------------------
    # 3. 画布布局初始化 (🌟 更改为 2 行 3 列)
    # --------------------------------------------------------
    # figsize 从 (28, 5.5) 改为 (18, 11) 以适应 2x3 的纵横比
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    # 提取当前 cmap 的两端颜色用于好坏标签图
    colormap_ref = cm.get_cmap(cmap)
    color_bad = colormap_ref(0.0)   # 坏芯片颜色 (极小值/冷色)
    color_good = colormap_ref(1.0)  # 好芯片颜色 (极大值/暖色)

    # 辅助函数：绘制连续趋势子图 (GPR)
    def draw_gpr_subplot(ax, zz_data, subplot_title, cbar_label):
        image = ax.imshow(
            zz_data,
            extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            origin='lower',
            cmap=cmap,
            aspect='equal'
        )
        draw_die_grid(ax, int(x_die.min()), int(x_die.max()), int(y_die.min()), int(y_die.max()))
        wafer_circle = Circle((-0.5, -0.5), wafer_radius, fill=False, color='black', linewidth=2)
        ax.add_patch(wafer_circle)
        
        ax.scatter(x_die, y_die, c='black', s=5, alpha=0.5)
        ax.set_title(subplot_title, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cbar = fig.colorbar(image, ax=ax, shrink=0.75)
        cbar.set_label(cbar_label)

    # 🌟 [第一行] 图1、图2、图3
    # 图1：Space (vl)
    draw_gpr_subplot(axes[0, 0], zz_vl, 'Space (vl) Spatial Trend\n(GPR Continuous)', 'Space Measurement (nm)')

    # 图2：MS (ll)
    draw_gpr_subplot(axes[0, 1], zz_ll, 'MS (ll) Spatial Trend\n(GPR Continuous)', 'MS Measurement (nm)')

    # 图3：预测寿命 t (矢量矩形块连续图)
    ax_t = axes[0, 2]
    norm = LogNorm(vmin=np.nanmin(t_scores), vmax=np.nanmax(t_scores)) if use_log_scale_for_t else plt.Normalize(vmin=np.nanmin(t_scores), vmax=np.nanmax(t_scores))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    for x, y, score in zip(x_die, y_die, t_scores):
        if np.isfinite(score):
            rect = Rectangle(
                xy=(x - 0.5, y - 0.5),
                width=1.0, height=1.0,
                facecolor=mapper.to_rgba(score),
                edgecolor='none', zorder=2
            )
            ax_t.add_patch(rect)

    ax_t.set_xlim(xx.min(), xx.max())
    ax_t.set_ylim(yy.min(), yy.max())
    draw_die_grid(ax_t, int(x_die.min()), int(x_die.max()), int(y_die.min()), int(y_die.max()))
    ax_t.add_patch(Circle((-0.5, -0.5), wafer_radius, fill=False, color='black', linewidth=2))
    ax_t.set_title('Predicted Lifetime (t)\n(Per-Die Vector Grid)', fontsize=14)
    ax_t.set_xlabel('X')
    ax_t.set_ylabel('Y')
    ax_t.set_aspect('equal')
    cbar = fig.colorbar(mapper, ax=ax_t, shrink=0.75)
    cbar.set_label('Reliability Lifetime Score (t)')

    # --------------------------------------------------------
    # 🌟 新增辅助函数：用于绘制二分类标签图
    # --------------------------------------------------------
    def draw_binary_label_subplot(ax, labels, subplot_title):
        for x, y, label in zip(x_die, y_die, labels):
            if np.isfinite(label):
                current_color = color_bad if int(label) == 0 else color_good
                rect = Rectangle(
                    xy=(x - 0.5, y - 0.5), 
                    width=1.0, height=1.0,
                    facecolor=current_color,
                    edgecolor='none',
                    zorder=2
                )
                ax.add_patch(rect)
                
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        draw_die_grid(ax, int(x_die.min()), int(x_die.max()), int(y_die.min()), int(y_die.max()))
        ax.add_patch(Circle((-0.5, -0.5), wafer_radius, fill=False, color='black', linewidth=2))
        ax.set_title(subplot_title, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        
        # 绘制离散型图例
        legend_elements = [
            Patch(facecolor=color_bad, edgecolor='black', label='Bad Die (0 / Intercept)'),
            Patch(facecolor=color_good, edgecolor='black', label='Good Die (1 / Pass)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)

    # 🌟 [第二行] 图4、图5、以及隐藏的空图
    # 图4：真实标签
    draw_binary_label_subplot(axes[1, 0], y_true, 'Ground Truth Reliability\n(Binary Labels)')

    # 图5：预测标签
    draw_binary_label_subplot(axes[1, 1], y_pred, 'Model Predicted Decision\n(Binary Classification)')

    # 图6：彻底隐藏不需要的右下角空坐标轴
    axes[1, 2].axis('off')

    # --------------------------------------------------------
    # 4. 调整间距与保存
    # --------------------------------------------------------
    fig.tight_layout(w_pad=2.0, h_pad=2.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved to: {output_path}')
        plt.close(fig)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config-path', type=str, default=None, help='Path to load a config')
    parser.add_argument('--pipeline-type', type=str, choices=['GPR', 'DPM'], help='Prediction Pipeline Type')
    parser.add_argument('--model-type', type=str, default=None, choices=['PowerLaw', 'SqrtE', 'InverseE'], help='Only for DPM pipelines')
    parser.add_argument('--M-structures', type=int, default=None, help='Parameter: M structures')
    parser.add_argument('--F-target', type=float, default=None, help='Parameter: F target')
    parser.add_argument('--N-samples-per-dim', type=int, default=None, help='Parameter: N samples per dim')
    parser.add_argument('--threshold', type=float, default=None, help='Parameter: threshold')
    parser.add_argument('--test-path', type=str, help='Path for test set')
    parser.add_argument('--save-path', type=str, help='Output path for test results')
    parser.add_argument('--fscore-beta', type=float, default=2.0, help='The beta parameter of F score')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size of wafers')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of CPUs')
    args = parser.parse_args()
    if args.pipeline_type == 'GPR': args.model_type = 'GPR'
    print(args)

    test_jobs, test_labels = build_test_job_and_label(args.test_path, args.save_path)
    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
        "pipeline_type": None,
        "M_structures": None,
        "F_target": None,
        "N_samples_per_dim": None,
        "threshold": None,
        "unit_model_kwargs":\
            {
                "model_type": None
            }
        }
    if args.pipeline_type is not None: config["pipeline_type"] = args.pipeline_type
    if args.M_structures is not None: config["M_structures"] = args.M_structures
    if args.F_target is not None: config["F_target"] = args.F_target
    if args.N_samples_per_dim is not None: config["N_samples_per_dim"] = args.N_samples_per_dim
    if args.threshold is not None: config["threshold"] = args.threshold
    if config["pipeline_type"] == 'GPR':
        config["unit_model_kwargs"]["actual_vl_max"] = 40.0
        config["unit_model_kwargs"]["actual_ll_max"] = 20.0

    results = batch_process_by_path(
        jobs_list=test_jobs,
        pipeline_config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    job_path = generate_and_create_mirrored_paths(args.test_path, args.save_path)
    for idx, (ttf_res, label_res) in enumerate(results):
        vl_matrix = load_matrix_csv(test_jobs[idx]['vl_path'])
        ll_matrix = load_matrix_csv(test_jobs[idx]['ll_path'])
        x, y, vl_val, _ = matrix_to_sparse_points(vl_matrix)
        _, _, ll_val, _ = matrix_to_sparse_points(ll_matrix)

        plot_five_trends(
            x, 
            y, 
            vl_val,
            ll_val, 
            ttf_res, 
            test_labels[idx],
            label_res,
            title='Wafer Spatial Trends, Lifetime & Binary Classification',
            cmap='turbo',
            output_path=job_path[idx][1] / 'wafer_maps.png',
            use_log_scale_for_t=True)


if __name__ == "__main__":
	main()