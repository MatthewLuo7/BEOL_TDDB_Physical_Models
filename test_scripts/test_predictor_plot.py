import matplotlib.cm as cm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LogNorm

from test_scripts.wafer_plot import create_wafer_mask, draw_die_grid, perform_gpr, load_matrix_csv, matrix_to_sparse_points
# Assume create_wafer_mask, draw_die_grid, perform_gpr are already defined 
# exactly as in your original wafer_plot.py

def plot_three_trends(
    x_die, 
    y_die, 
    vl_val, 
    ll_val, 
    t_scores, 
    title='Wafer Spatial Trends & Predicted Lifetime',
    cmap='turbo',
    output_path=None,
    use_log_scale_for_t=True
):
    """
    绘制三合一晶圆图：
    - 左/中：Space 和 MS 的连续 GPR 空间趋势插值图
    - 右：预测寿命 t 的矢量方格图 (通过 Rectangle 完美对齐网格线，彻底消灭对齐缺陷)
    """
    # --------------------------------------------------------
    # 1. 工艺尺寸 vl 和 ll 的连续空间 GPR 插值
    # --------------------------------------------------------
    print("Performing GPR for Space (vl)...")
    xx, yy, zz_vl, _ = perform_gpr(x_die, y_die, vl_val)
    
    print("Performing GPR for MS (ll)...")
    _, _, zz_ll, _ = perform_gpr(x_die, y_die, ll_val)

    # --------------------------------------------------------
    # 2. 晶圆掩膜 (仅用于左、中两张连续插值图)
    # --------------------------------------------------------
    valid_distance = np.sqrt(x_die**2 + y_die**2)
    wafer_radius = valid_distance.max()

    wafer_mask = create_wafer_mask(xx + 0.5, yy + 0.5, wafer_radius)
    zz_vl = np.where(wafer_mask, zz_vl, np.nan)
    zz_ll = np.where(wafer_mask, zz_ll, np.nan)

    # --------------------------------------------------------
    # 3. 画布布局初始化 (1行3列)
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)

    # 辅助函数：绘制左/中两张连续趋势子图
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

    # 左图：Space
    draw_gpr_subplot(axes[0], zz_vl, 'Space (vl) Spatial Trend\n(GPR Continuous)', 'Space Measurement (nm)')

    # 中图：MS
    draw_gpr_subplot(axes[1], zz_ll, 'MS (ll) Spatial Trend\n(GPR Continuous)', 'MS Measurement (nm)')

    # ========================================================
    # 右图：预测寿命 (t) —— 采用【矢量矩形块】无缝填充
    # ========================================================
    ax = axes[2]
    
    # 初始化映射颜色的归一化工具 (ScalarMappable)
    norm = LogNorm(vmin=np.nanmin(t_scores), vmax=np.nanmax(t_scores)) if use_log_scale_for_t else plt.Normalize(vmin=np.nanmin(t_scores), vmax=np.nanmax(t_scores))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # 【核心优化】：遍历每个有效 Die，直接绘制完全契合其物理边界的矢量 Rectangle
    for x, y, score in zip(x_die, y_die, t_scores):
        if np.isfinite(score):
            rect = Rectangle(
                xy=(x - 0.5, y - 0.5), # 左下角坐标，与网格线精确重合
                width=1.0,             # 每个 Die 的宽度为 1
                height=1.0,            # 每个 Die 的高度为 1
                facecolor=mapper.to_rgba(score),
                edgecolor='none',      # 边缘留给 draw_die_grid 去画，避免重复叠加
                zorder=2               # 置于网格线下方
            )
            ax.add_patch(rect)

    # 强制让右图的坐标轴显示范围与左边两张 GPR 插值图完全一致，实现完美视觉对齐
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    # 绘制方格网格线 (将以矢量级精度完美卡死色块边界)
    draw_die_grid(ax, int(x_die.min()), int(x_die.max()), int(y_die.min()), int(y_die.max()))
    
    # 绘制晶圆外圈圆
    wafer_circle = Circle((-0.5, -0.5), wafer_radius, fill=False, color='black', linewidth=2)
    ax.add_patch(wafer_circle)

    ax.set_title('Predicted Lifetime (t)\n(Per-Die Vector Grid)', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    # 为矢量图绑定对应的 Colorbar
    cbar = fig.colorbar(mapper, ax=ax, shrink=0.75)
    cbar.set_label('Reliability Lifetime Score (t)')

    # --------------------------------------------------------
    # 4. 调整间距与保存
    # --------------------------------------------------------
    fig.tight_layout(w_pad=2.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved to: {output_path}')
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    from src.unit_level_models.model_factory import unit_model_factory
    from src.prediction_pipeline import IntegratedWaferPipeline

    # 1. 加载并配置预测器
    # -------- gpr -------
    config = {
    "pipeline_type": "GPR",
    "M_structures": 1_000_000,
    "F_target": 1e-4,
    "N_samples_per_dim": 32,
    "unit_model_kwargs": {
        "model_type": "GPR",
        "actual_vl_max":40.0,
        "actual_ll_max":20.0
        }
    }

    # # ------- dpm -------
    # config = {
    # "pipeline_type": "DPM",
    # "M_structures": 1_000_000,
    # "F_target": 1e-4,
    # "N_samples_per_dim": 32,
    # "unit_model_kwargs": {
    #     "model_type": "PowerLaw"    # PowerLaw, SqrtE, InverseE
    #     }
    # }

    predictor = IntegratedWaferPipeline(config=config)

    # 2. 读取实际的晶圆测量矩阵 CSV 文件
    lot_idx = '001'
    wafer_idx = '24'
    vl_matrix = load_matrix_csv(f'../BEOL_TDDB_D2/data/lot_{lot_idx}/csv/wafer_{wafer_idx}/Space.csv')
    ll_matrix = load_matrix_csv(f'../BEOL_TDDB_D2/data/lot_{lot_idx}/csv/wafer_{wafer_idx}/MS.csv')
    
    # 将矩阵转换为稀疏点集 (Die的坐标与对应的测量值)
    x, y, vl_val = matrix_to_sparse_points(vl_matrix)
    _, _, ll_val = matrix_to_sparse_points(ll_matrix)

    # 3. 执行端到端的 Die 级别可靠性预测
    t_scores = predictor.predict_ttf(
        x_die=x, y_die=y, 
        vl_data=vl_val, ll_data=ll_val)

    # 4. 绘制三合一的晶圆空间趋势分布图 (Space, MS, 预测寿命 t)
    plot_three_trends(
        x_die=x, 
        y_die=y, 
        vl_val=vl_val, 
        ll_val=ll_val, 
        t_scores=t_scores, 
        output_path=None, # 指定保存路径
        use_log_scale_for_t=True  # 寿命跨度通常很大，强烈推荐使用对数坐标系
    )
    
    print("Workflow completely successfully!")

    # ----------------------- prev code -----------------------

    # # # ----------- GPR ----------
    # # # 1. 加载并配置 Weibull GPR 模型 (自动处理实际输入的 Scale 映射)
    # # from src.wafer_level_models.wafer_reliability_engines import GPRWaferEngine
    # # weibull_model = unit_model_factory(model_type='GPR', actual_vl_max=40.0, actual_ll_max=20.0)

    # # # 2. 实例化晶圆可靠性预测器
    # # # 假设每个 Die 有 1,000,000 个结构，目标失效概率为 0.01% (1e-4)
    # # predictor = GPRWaferEngine(
    # #     unit_model_joint=weibull_model, 
    # #     M_structures=1_000_000, 
    # #     F_target=1e-4
    # # )

    # # # ----------- DPM Models ----------
    # # 1. 加载并配置 Weibull GPR 模型 (自动处理实际输入的 Scale 映射)
    # from src.wafer_level_models.wafer_reliability_engines import DPMWaferEngine

    # model_type = 'PowerLaw' # SqrtE, InverseE
    # weibull_model_via = unit_model_factory(model_type, target_component='via')
    # weibull_model_line = unit_model_factory(model_type, target_component='line')

    # # 2. 实例化晶圆可靠性预测器
    # # 假设每个 Die 有 1,000,000 个结构，目标失效概率为 0.01% (1e-4)
    # predictor = DPMWaferEngine(
    #     unit_model_via=weibull_model_via,
    #     unit_model_line=weibull_model_line,
    #     M_structures=1_000_000, 
    #     F_target=1e-4
    # )

    # # 3. 读取实际的晶圆测量矩阵 CSV 文件
    # lot_idx = '001'
    # wafer_idx = '24'
    # vl_matrix = load_matrix_csv(f'../BEOL_TDDB_D2/data/lot_{lot_idx}/csv/wafer_{wafer_idx}/Space.csv')
    # ll_matrix = load_matrix_csv(f'../BEOL_TDDB_D2/data/lot_{lot_idx}/csv/wafer_{wafer_idx}/MS.csv')
    
    # # 将矩阵转换为稀疏点集 (Die的坐标与对应的测量值)
    # x, y, vl_val = matrix_to_sparse_points(vl_matrix)
    # _, _, ll_val = matrix_to_sparse_points(ll_matrix)

    # # 4. 执行端到端的 Die 级别可靠性预测
    # t_scores = predictor.predict_die_lifetimes(
    #     x_die=x, y_die=y, 
    #     vl_data=vl_val, ll_data=ll_val, 
    #     N_samples_per_dim=32  # 意味着每个 Die 内部产生 32x32=1024 个抽样点
    # )

    # # 5. 绘制三合一的晶圆空间趋势分布图 (Space, MS, 预测寿命 t)
    # plot_three_trends(
    #     x_die=x, 
    #     y_die=y, 
    #     vl_val=vl_val, 
    #     ll_val=ll_val, 
    #     t_scores=t_scores, 
    #     output_path=None, # 指定保存路径
    #     use_log_scale_for_t=True  # 寿命跨度通常很大，强烈推荐使用对数坐标系
    # )
    
    # print("Workflow completely successfully!")