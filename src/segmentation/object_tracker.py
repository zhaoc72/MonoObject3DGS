"""
Object Tracker
跨帧物体追踪，确保物体ID的一致性
"""

import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from collections import deque


class ObjectTracker:
    """
    基于特征和IoU的多物体追踪器
    确保跨帧物体ID的一致性
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        feature_threshold: float = 0.7
    ):
        """
        Args:
            max_age: 物体消失后保持的最大帧数
            min_hits: 确认物体需要的最小匹配次数
            iou_threshold: IoU匹配阈值
            feature_threshold: 特征相似度阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        self.tracks = []  # 活跃的轨迹
        self.next_id = 0  # 下一个可用ID
        self.frame_count = 0
        
        print(f"✓ ObjectTracker初始化: max_age={max_age}, min_hits={min_hits}")
        
    def update(
        self,
        masks: List[Dict],
        features: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        更新追踪器
        
        Args:
            masks: 当前帧的mask列表，每个包含:
                - segmentation: (H, W) bool mask
                - bbox: [x, y, w, h]
                - area: int
            features: 可选的特征向量 (N, D)
            
        Returns:
            tracked_masks: 带有ID的mask列表
        """
        self.frame_count += 1
        
        # 如果没有轨迹，初始化所有检测为新轨迹
        if len(self.tracks) == 0:
            for i, mask in enumerate(masks):
                self._initiate_track(mask, features[i] if features is not None else None)
            return self._get_confirmed_tracks()
        
        # 预测当前位置（简单的恒速模型）
        for track in self.tracks:
            track['predicted_bbox'] = self._predict_bbox(track)
        
        # 计算匹配矩阵
        cost_matrix = self._compute_cost_matrix(masks, features)
        
        # 使用匈牙利算法进行匹配
        if cost_matrix.size > 0:
            matched_indices = self._match(cost_matrix)
        else:
            matched_indices = []
        
        # 更新匹配的轨迹
        matched_detections = set()
        matched_tracks = set()
        
        for track_idx, det_idx in matched_indices:
            track = self.tracks[track_idx]
            mask = masks[det_idx]
            feature = features[det_idx] if features is not None else None
            
            self._update_track(track, mask, feature)
            matched_detections.add(det_idx)
            matched_tracks.add(track_idx)
        
        # 处理未匹配的轨迹
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks:
                track['age'] += 1
                track['hits'] = 0
        
        # 删除过时的轨迹
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        # 为未匹配的检测创建新轨迹
        for det_idx, mask in enumerate(masks):
            if det_idx not in matched_detections:
                feature = features[det_idx] if features is not None else None
                self._initiate_track(mask, feature)
        
        return self._get_confirmed_tracks()
    
    def _compute_cost_matrix(
        self,
        masks: List[Dict],
        features: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """计算轨迹和检测之间的代价矩阵"""
        if len(self.tracks) == 0 or len(masks) == 0:
            return np.empty((0, 0))
        
        cost_matrix = np.zeros((len(self.tracks), len(masks)))
        
        for i, track in enumerate(self.tracks):
            for j, mask in enumerate(masks):
                # IoU代价
                iou = self._compute_iou(
                    track['bbox'],
                    mask['bbox']
                )
                iou_cost = 1.0 - iou
                
                # 特征代价
                if features is not None and track['feature'] is not None:
                    feature_sim = torch.cosine_similarity(
                        track['feature'].unsqueeze(0),
                        features[j].unsqueeze(0)
                    ).item()
                    feature_cost = 1.0 - feature_sim
                else:
                    feature_cost = 0.0
                
                # 组合代价
                cost_matrix[i, j] = 0.6 * iou_cost + 0.4 * feature_cost
        
        return cost_matrix
    
    def _match(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """使用匈牙利算法进行最优匹配"""
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 0.7:  # 只保留低代价的匹配
                matches.append((row, col))
        
        return matches
    
    def _initiate_track(self, mask: Dict, feature: Optional[torch.Tensor] = None):
        """初始化新轨迹"""
        track = {
            'id': self.next_id,
            'bbox': mask['bbox'],
            'mask': mask['segmentation'],
            'area': mask['area'],
            'feature': feature,
            'age': 0,
            'hits': 1,
            'history': deque(maxlen=10),
            'velocity': np.array([0.0, 0.0, 0.0, 0.0])  # [dx, dy, dw, dh]
        }
        track['history'].append(mask['bbox'])
        
        self.tracks.append(track)
        self.next_id += 1
    
    def _update_track(self, track: Dict, mask: Dict, feature: Optional[torch.Tensor] = None):
        """更新轨迹"""
        # 更新边界框
        old_bbox = track['bbox']
        new_bbox = mask['bbox']
        
        # 计算速度（简单差分）
        track['velocity'] = np.array(new_bbox) - np.array(old_bbox)
        
        # 更新信息
        track['bbox'] = new_bbox
        track['mask'] = mask['segmentation']
        track['area'] = mask['area']
        track['age'] = 0
        track['hits'] += 1
        track['history'].append(new_bbox)
        
        # 更新特征（指数移动平均）
        if feature is not None:
            if track['feature'] is not None:
                alpha = 0.7
                track['feature'] = alpha * track['feature'] + (1 - alpha) * feature
            else:
                track['feature'] = feature
    
    def _predict_bbox(self, track: Dict) -> List[float]:
        """预测下一帧的边界框位置"""
        # 简单的恒速模型
        predicted = np.array(track['bbox']) + track['velocity']
        return predicted.tolist()
    
    def _get_confirmed_tracks(self) -> List[Dict]:
        """返回确认的轨迹"""
        confirmed = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits:
                confirmed.append({
                    'id': track['id'],
                    'segmentation': track['mask'],
                    'bbox': track['bbox'],
                    'area': track['area'],
                })
        return confirmed
    
    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 转换为 (x1, y1, x2, y2) 格式
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # 计算交集
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_track_info(self) -> Dict:
        """获取追踪器信息"""
        return {
            'active_tracks': len(self.tracks),
            'confirmed_tracks': len([t for t in self.tracks if t['hits'] >= self.min_hits]),
            'next_id': self.next_id,
            'frame_count': self.frame_count
        }
    
    def reset(self):
        """重置追踪器"""
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0


if __name__ == "__main__":
    # 测试代码
    print("=== 测试ObjectTracker ===")
    
    tracker = ObjectTracker(max_age=30, min_hits=3)
    
    # 模拟多帧追踪
    for frame_id in range(5):
        print(f"\n帧 {frame_id}:")
        
        # 模拟检测结果
        num_objects = np.random.randint(2, 5)
        masks = []
        features = []
        
        for i in range(num_objects):
            x, y = np.random.randint(0, 400), np.random.randint(0, 400)
            w, h = np.random.randint(50, 100), np.random.randint(50, 100)
            
            mask = {
                'segmentation': np.zeros((512, 512), dtype=bool),
                'bbox': [x, y, w, h],
                'area': w * h
            }
            mask['segmentation'][y:y+h, x:x+w] = True
            
            masks.append(mask)
            features.append(torch.randn(768))  # 模拟DINOv2特征
        
        features = torch.stack(features)
        
        # 更新追踪器
        tracked = tracker.update(masks, features)
        
        print(f"  检测: {len(masks)} 个物体")
        print(f"  追踪: {len(tracked)} 个确认物体")
        
        for obj in tracked:
            print(f"    ID {obj['id']}: bbox={obj['bbox']}")
    
    # 显示追踪器状态
    info = tracker.get_track_info()
    print(f"\n追踪器状态:")
    print(f"  活跃轨迹: {info['active_tracks']}")
    print(f"  确认轨迹: {info['confirmed_tracks']}")
    print(f"  总共分配ID: {info['next_id']}")
    
    print("\n测试完成！")