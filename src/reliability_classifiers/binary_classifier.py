import numpy as np

# ==============================================================================
# 1. Standalone Metric Function / 独立的评估指标函数（方便后续扩展为多分类）
# ==============================================================================
def calculate_f_beta_score(y_true: np.ndarray, y_pred: np.ndarray, 
                             beta: float = 2.0, pos_label: int = 0) -> float:
    """
    Calculate the F-beta score for binary classification.
    Designed for highly imbalanced semiconductor data where catching rare failures is critical.
    
    计算二分类的 F-beta 分数。
    专为极度不平衡的半导体数据设计，其中捕获罕见的失效芯片（Bad Die）是核心任务。
    
    Args:
        y_true: Ground truth binary labels (0 for Fail/Bad, 1 for Pass/Good).
                真实二分类标签（0 代表拦截/坏芯片，1 代表放行/好芯片）。
        y_pred: Predicted binary labels from the classifier.
                分类器输出的预测二分类标签。
        beta: Weight of Recall relative to Precision. 
              If beta = 2, Recall is 2x more important than Precision (penalizes Escapes).
              Recall 相对于 Precision 的权重。若 beta=2，则 Recall 的重要性是 Precision 的 2 倍（严厉惩罚漏检）。
        pos_label: The class label defined as "Positive" (Target of interest).
                   Defaul is 0 because Low Reliability (Fail) is what we must capture.
                   被定义为“正类”的标签（我们核心关注的目标）。
                   默认是 0，因为低可靠性（Fail）芯片是我们必须要抓出来的。
                   
    Returns:
        float: The calculated F-beta score between 0.0 and 1.0.
    """
    # Vectorized extraction of TP, FP, FN based on the specified pos_label
    # 基于指定的正类标签，通过向量化操作提取 TP, FP, FN
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label)) # Overkill / 误杀
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label)) # Escape / 漏检
    
    # Calculate Precision and Recall with zero-division protection
    # 计算 Precision 和 Recall，带有防零除保护
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F-beta score / 计算 F-beta 分数
    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    
    if denominator == 0:
        return 0.0
        
    f_beta = (1 + beta_sq) * (precision * recall) / denominator
    return float(f_beta)


# ==============================================================================
# 2. Layer 3 Classifier Class / 第三层分类器类
# ==============================================================================
class BinaryReliabilityClassifier:
    """
    Layer 3 Classifier: Maps continuous Time-to-Failure (TTF) into binary decisions.
    Class 0: Low Reliability / Bad Die (TTF < threshold) -> Intercept
    Class 1: High Reliability / Good Die (TTF >= threshold) -> Pass
    
    第三层分类器：将连续的失效时间 (TTF) 映射为二分类决策。
    分类 0：低可靠性 / 坏芯片 (TTF < 阈值) -> 拦截
    分类 1：高可靠性 / 好芯片 (TTF >= 阈值) -> 放行
    """
    def __init__(self, threshold: float):
        """
        Args:
            threshold: The cutoff lifetime separating bad and good dies.
                       区分好坏芯片的寿命截断阈值。
        """
        self.threshold = threshold

    def classify(self, ttf_array: np.ndarray) -> np.ndarray:
        """
        Execute binary thresholding. / 执行二分类阈值切分。
        """
        # Returns 1 (Good) if ttf >= threshold, else 0 (Bad)
        # 如果 ttf >= 阈值则返回 1（好），否则返回 0（坏）
        return np.where(ttf_array >= self.threshold, 1, 0)