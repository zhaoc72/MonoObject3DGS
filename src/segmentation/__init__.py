"""
Segmentation Module
语义感知的物体分割模块
"""

from .dinov2_extractor import DINOv2Extractor
from .sam_segmenter import SAMSegmenter
from .object_tracker import ObjectTracker
from .semantic_classifier import SemanticClassifier

__all__ = [
    'DINOv2Extractor',
    'SAMSegmenter',
    'ObjectTracker',
    'SemanticClassifier',
]

__version__ = '0.1.0'