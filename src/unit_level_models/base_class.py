import numpy as np
import pathlib
from abc import ABC, abstractmethod

# ==============================================================================
# Unified Base Model Abstract Class / 统一基础模型抽象基类
# ==============================================================================
class BaseTDDBModel(ABC):
    """
    Abstract base class for TDDB base models.
    All underlying reliability models (GPR or physics-based) must inherit from 
    this class and implement a unified interface.
    
    TDDB 基础模型的抽象基类。
    所有底层可靠性模型（无论是基于GPR插值还是纯物理推导）都必须继承此类并实现统一的接口。
    """
    @abstractmethod
    def predict_weibull_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass