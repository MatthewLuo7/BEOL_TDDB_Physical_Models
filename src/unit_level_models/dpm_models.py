import numpy as np
from abc import abstractmethod

from src.unit_level_models.base_class import BaseTDDBModel

V_OP = 0.7  # Fallback default value / 兜底默认值

# ==============================================================================
# Physics Base Model Parent Class: Base DPM / 物理基础模型母类：Base DPM
# ==============================================================================
class BaseDPMModel(BaseTDDBModel):
    """
    Base class for Dynamic Percolation Model (DPM) physics.
    Upgraded to support unified 2D inputs [vl_space, ll_space] to perfectly align with GPR.
    
    动态渗透模型 (DPM) 物理公式的基类。
    升级后支持统一的 2维输入 [vl_space, ll_space]，从而与 GPR 模型实现完美对齐。
    """
    def __init__(self, target_component: str = "line", v_op: float = V_OP):
        """
        Args:
            target_component (str): "via" or "line". Determines which spacing column to route.
                                    "via" 或 "line"。决定自动路由并提取哪一列间距数据。
            v_op (float): Operating voltage. 
                          工作电压。
        """
        self.target_component = target_component.lower()
        assert self.target_component in ["via", "line"], "target_component must be 'via' or 'line'"
        
        self.v_op = v_op
        
        # Table 1 Shared Physical Constants / Table 1 共有物理常数
        self.C_sc = 2.134   # Weibull slope coefficient / Weibull 斜率系数
        self.P_sc = 0.847   # Weibull slope exponent / Weibull 斜率指数
        self.W_mu = -0.3665 # Scale parameter for DOT distribution / DOT 分布的尺度参数
        self.mu_sc = 0.914  # Scale factor for eta_DOT / eta_DOT 的缩放因子

    def _extract_spacing(self, inputs: np.ndarray) -> np.ndarray:
        """
        Feature Routing Layer: Extract the corresponding 1D spacing based on target_component.
        Supports both unified 2D matrix (n_samples, 2) and fallback 1D array (n_samples,).
        
        特征路由层：根据目标组件自动提取对应的 1维间距数组。
        同时支持统一的 2维特征矩阵 (n_samples, 2) 和向后兼容的 1维数组 (n_samples,)。
        """
        arr = np.asarray(inputs, dtype=float)
        
        if arr.ndim == 2 and arr.shape[1] == 2:
            # Column 0: vl_space (via spacing), Column 1: ll_space (line spacing / MS)
            # 第 0 列：vl_space (via间距)，第 1 列：ll_space (line间距 / MS)
            if self.target_component == "via":
                return arr[:, 0]
            else:
                return arr[:, 1]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            return arr.flatten()
        
        return arr

    def calc_beta_DOT(self, S: np.ndarray) -> np.ndarray:
        """Equation (2): Calculates the Weibull slope of the DOT distribution."""
        """公式 (2): 计算 DOT 分布的 Weibull 斜率。"""
        return self.C_sc * (S ** self.P_sc)

    def calc_eta_DOT(self, S: np.ndarray) -> np.ndarray:
        """Equation (3): Calculates the local Weibull scale parameter for DOT distribution."""
        """公式 (3): 计算 DOT 分布的局部 Weibull 尺度参数。"""
        beta_DOT = self.calc_beta_DOT(S)
        return self.mu_sc * np.exp(-self.W_mu / beta_DOT)

    def calc_Es(self, S: np.ndarray) -> np.ndarray:
        """Equation (17): Calculates the characteristic E-field (Es) for a given spacing S."""
        """公式 (17): 计算给定间距 S 下的特征电场 (Es)。"""
        return 1.0 / (0.85 + 2.91 / S)

    def calc_m(self, S: np.ndarray) -> np.ndarray:
        """Equation (18): Calculates the spacing-dependent exponent m for DOT-tBD relation."""
        """公式 (18): 计算 DOT-tBD 关系的间距相关指数 m。"""
        return 0.5659 * (S ** (-0.455))

    def calc_beta_tBD(self, S: np.ndarray) -> np.ndarray:
        """Calculates the local Weibull shape parameter beta for time-to-breakdown."""
        """计算最终的 Time-to-breakdown Weibull 形状参数 beta。"""
        return self.calc_m(S) * self.calc_beta_DOT(S)

    @abstractmethod
    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        """Calculate the natural log of the final scale parameter: ln(eta_tBD)."""
        """计算最终的弹性尺度参数的自然对数 ln(eta_tBD)。"""
        pass

    def predict_weibull_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Unified interface implementation: seamlessly process 2D/1D inputs and return parameters.
        
        统一接口实现：无缝处理 2D/1D 输入，并返回对应的 Weibull 参数。
        """
        # Automatically route and extract the correct physical spacing
        # 自动路由并提取出正确的物理间距
        S = self._extract_spacing(inputs)
        
        beta_tBD = self.calc_beta_tBD(S)
        ln_eta_tBD = self.calc_ln_eta_tBD(S)
        
        # Secure handling to prevent float64 overflow
        # 安全防溢出处理
        eta_tBD = np.full_like(ln_eta_tBD, np.inf, dtype=float)
        safe_mask = ln_eta_tBD <= 709.0  
        eta_tBD[safe_mask] = np.exp(ln_eta_tBD[safe_mask])
        
        # If the original input was a 2D matrix, reshape outputs to (n_samples, 1) for consistency
        # 如果原始输入是 2D 矩阵，将输出重塑为 (n_samples, 1) 以保持形状一致性
        if inputs.ndim == 2:
            beta_tBD = beta_tBD.reshape(-1, 1)
            eta_tBD = eta_tBD.reshape(-1, 1)
            
        return beta_tBD, eta_tBD


# ==============================================================================
# Derived Subclasses (Automatically Inherit the Routing Feature)
#    派生子类（自动继承特征路由功能）
# ==============================================================================

class PowerLawDPMModel(BaseDPMModel):
    """1) Classical Power-Law Acceleration Model. / 1) 经典幂律加速模型。"""
    def calc_ma(self, S: np.ndarray) -> np.ndarray:
        return 20.66 / np.tanh(0.073 * S)

    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        eta_DOT = self.calc_eta_DOT(S)
        E_local = self.v_op / S
        m_val = self.calc_m(S)
        Es_val = self.calc_Es(S)
        ma_val = self.calc_ma(S)
        
        ln_AF = -ma_val * np.log(E_local / Es_val)
        ln_eta_supercell = ln_AF + (1.0 / m_val) * np.log(eta_DOT)
        
        beta_tBD = self.calc_beta_tBD(S)
        return ln_eta_supercell + (1.0 / beta_tBD) * np.log(S ** 2)


class SqrtEDPMModel(BaseDPMModel):
    """2) Sqrt(E) Acceleration Model. / 2) Sqrt(E) 电场加速模型。"""
    def __init__(self, target_component: str = "line", v_op: float = V_OP):
        super().__init__(target_component, v_op)
        self.gamma_coeff = 49.14

    def calc_gamma_sqrtE(self, S: np.ndarray) -> np.ndarray:
        return self.gamma_coeff / np.tanh(0.073 * S)

    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        eta_DOT = self.calc_eta_DOT(S)
        E_local = self.v_op / S
        m_val = self.calc_m(S)
        Es_val = self.calc_Es(S)
        gamma = self.calc_gamma_sqrtE(S)
        
        ln_AF = gamma * (np.sqrt(Es_val) - np.sqrt(E_local))
        ln_eta_supercell = ln_AF + (1.0 / m_val) * np.log(eta_DOT)
        
        beta_tBD = self.calc_beta_tBD(S)
        return ln_eta_supercell + (1.0 / beta_tBD) * np.log(S ** 2)


class InverseEDPMModel(BaseDPMModel):
    """3) 1/E Acceleration Model. / 3) 1/E 电场加速模型。"""
    def __init__(self, target_component: str = "line", v_op: float = V_OP):
        super().__init__(target_component, v_op)
        self.G_1E_coeff = 14.39

    def calc_G_1E(self, S: np.ndarray) -> np.ndarray:
        return self.G_1E_coeff / np.tanh(0.073 * S)

    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        eta_DOT = self.calc_eta_DOT(S)
        E_local = self.v_op / S
        m_val = self.calc_m(S)
        Es_val = self.calc_Es(S)
        G_val = self.calc_G_1E(S)
        
        ln_AF = G_val * (1.0 / E_local - 1.0 / Es_val)
        ln_eta_supercell = ln_AF + (1.0 / m_val) * np.log(eta_DOT)
        
        beta_tBD = self.calc_beta_tBD(S)
        return ln_eta_supercell + (1.0 / beta_tBD) * np.log(S ** 2)