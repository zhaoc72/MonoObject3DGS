"""
Shape Prior Module
形状先验模块 - 核心创新点
"""

from .explicit_prior import ExplicitShapePrior
from .implicit_prior import ImplicitShapePrior, ShapeEncoder, ShapeDecoder
from .prior_fusion import AdaptivePriorFusion, PriorConfig
from .regularizers import ShapePriorRegularizer

__all__ = [
    'ExplicitShapePrior',
    'ImplicitShapePrior',
    'ShapeEncoder',
    'ShapeDecoder',
    'AdaptivePriorFusion',
    'PriorConfig',
    'ShapePriorRegularizer',
]