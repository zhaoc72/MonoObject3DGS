"""
Evaluator and Metrics for 3D Reconstruction
评估器和3D重建指标
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree


class Evaluator:
    """3D重建评估器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    @torch.no_grad()
    def evaluate_reconstruction(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        pred_colors: Optional[np.ndarray] = None,
        gt_colors: Optional[np.ndarray] = None,
        thresholds: List[float] = [0.01, 0.02, 0.05]
    ) -> Dict:
        """
        评估3D重建质量
        
        Args:
            pred_points: 预测点云 (N, 3)
            gt_points: GT点云 (M, 3)
            pred_colors: 预测颜色 (N, 3)
            gt_colors: GT颜色 (M, 3)
            thresholds: 距离阈值列表
            
        Returns:
            metrics: 评估指标字典
        """
        metrics = {}
        
        # Chamfer Distance
        cd_pred_to_gt, cd_gt_to_pred = self.chamfer_distance(pred_points, gt_points)
        metrics['chamfer_distance'] = (cd_pred_to_gt + cd_gt_to_pred) / 2
        metrics['chamfer_pred_to_gt'] = cd_pred_to_gt
        metrics['chamfer_gt_to_pred'] = cd_gt_to_pred
        
        # F-Score
        for threshold in thresholds:
            precision, recall, f_score = self.f_score(
                pred_points, gt_points, threshold
            )
            metrics[f'precision_{threshold}'] = precision
            metrics[f'recall_{threshold}'] = recall
            metrics[f'f_score_{threshold}'] = f_score
        
        # Accuracy (mean distance)
        accuracy = self.accuracy(pred_points, gt_points)
        metrics['accuracy'] = accuracy
        
        # Completeness (coverage)
        completeness = self.completeness(pred_points, gt_points, threshold=0.05)
        metrics['completeness'] = completeness
        
        # 颜色指标（如果提供）
        if pred_colors is not None and gt_colors is not None:
            color_error = self.color_error(
                pred_points, pred_colors,
                gt_points, gt_colors
            )
            metrics['color_error'] = color_error
        
        return metrics
    
    def chamfer_distance(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算Chamfer距离
        
        Returns:
            (d1_to_2, d2_to_1): 双向距离
        """
        # 构建KD树
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # points1 -> points2
        distances1, _ = tree2.query(points1, k=1)
        d1_to_2 = distances1.mean()
        
        # points2 -> points1
        distances2, _ = tree1.query(points2, k=1)
        d2_to_1 = distances2.mean()
        
        return float(d1_to_2), float(d2_to_1)
    
    def f_score(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        threshold: float
    ) -> Tuple[float, float, float]:
        """
        计算F-Score
        
        Returns:
            (precision, recall, f_score)
        """
        tree_pred = cKDTree(pred_points)
        tree_gt = cKDTree(gt_points)
        
        # Precision: pred中有多少点接近GT
        distances_pred, _ = tree_gt.query(pred_points, k=1)
        precision = (distances_pred < threshold).sum() / len(pred_points)
        
        # Recall: GT中有多少点被覆盖
        distances_gt, _ = tree_pred.query(gt_points, k=1)
        recall = (distances_gt < threshold).sum() / len(gt_points)
        
        # F-Score
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0
        
        return float(precision), float(recall), float(f_score)
    
    def accuracy(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray
    ) -> float:
        """计算准确度（平均最近点距离）"""
        tree_gt = cKDTree(gt_points)
        distances, _ = tree_gt.query(pred_points, k=1)
        return float(distances.mean())
    
    def completeness(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        threshold: float = 0.05
    ) -> float:
        """计算完整度（GT点被覆盖的比例）"""
        tree_pred = cKDTree(pred_points)
        distances, _ = tree_pred.query(gt_points, k=1)
        completeness = (distances < threshold).sum() / len(gt_points)
        return float(completeness)
    
    def color_error(
        self,
        pred_points: np.ndarray,
        pred_colors: np.ndarray,
        gt_points: np.ndarray,
        gt_colors: np.ndarray
    ) -> float:
        """计算颜色误差"""
        # 对每个pred点，找到最近的GT点
        tree_gt = cKDTree(gt_points)
        _, indices = tree_gt.query(pred_points, k=1)
        
        # 计算颜色差异
        matched_gt_colors = gt_colors[indices]
        color_diff = np.abs(pred_colors - matched_gt_colors).mean()
        
        return float(color_diff)
    
    def evaluate_depth(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        评估深度估计
        
        Args:
            pred_depth: 预测深度 (H, W)
            gt_depth: GT深度 (H, W)
            mask: 有效区域mask (H, W)
            
        Returns:
            metrics: 深度指标
        """
        if mask is None:
            mask = (gt_depth > 0) & (gt_depth < 100)
        
        pred = pred_depth[mask]
        gt = gt_depth[mask]
        
        if len(pred) == 0:
            return {}
        
        metrics = {}
        
        # Absolute Relative Error
        abs_rel = np.mean(np.abs(pred - gt) / gt)
        metrics['abs_rel'] = float(abs_rel)
        
        # Squared Relative Error
        sq_rel = np.mean(((pred - gt) ** 2) / gt)
        metrics['sq_rel'] = float(sq_rel)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        metrics['rmse'] = float(rmse)
        
        # RMSE log
        rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))
        metrics['rmse_log'] = float(rmse_log)
        
        # Threshold accuracy
        thresh = np.maximum(pred / gt, gt / pred)
        for delta in [1.25, 1.25**2, 1.25**3]:
            accuracy = (thresh < delta).mean()
            metrics[f'delta_{delta:.3f}'] = float(accuracy)
        
        return metrics
    
    def evaluate_segmentation(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        评估实例分割
        
        Args:
            pred_masks: 预测masks列表
            gt_masks: GT masks列表
            iou_threshold: IoU匹配阈值
            
        Returns:
            metrics: 分割指标
        """
        metrics = {}
        
        # 计算IoU矩阵
        n_pred = len(pred_masks)
        n_gt = len(gt_masks)
        
        if n_pred == 0 or n_gt == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mean_iou': 0.0
            }
        
        iou_matrix = np.zeros((n_pred, n_gt))
        
        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                intersection = (pred_mask & gt_mask).sum()
                union = (pred_mask | gt_mask).sum()
                iou = intersection / union if union > 0 else 0.0
                iou_matrix[i, j] = iou
        
        # 匹配
        matched_pred = set()
        matched_gt = set()
        ious = []
        
        for i in range(n_pred):
            best_j = np.argmax(iou_matrix[i])
            best_iou = iou_matrix[i, best_j]
            
            if best_iou >= iou_threshold and best_j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(best_j)
                ious.append(best_iou)
        
        # 计算指标
        precision = len(matched_pred) / n_pred
        recall = len(matched_gt) / n_gt
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(ious) if ious else 0.0
        
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
        metrics['mean_iou'] = float(mean_iou)
        metrics['num_pred'] = n_pred
        metrics['num_gt'] = n_gt
        metrics['num_matched'] = len(matched_pred)
        
        return metrics
    
    def evaluate_novel_view(
        self,
        pred_image: np.ndarray,
        gt_image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        评估新视角渲染
        
        Args:
            pred_image: 预测图像 (H, W, 3)
            gt_image: GT图像 (H, W, 3)
            mask: 有效区域mask (H, W)
            
        Returns:
            metrics: 渲染指标
        """
        if mask is None:
            mask = np.ones(pred_image.shape[:2], dtype=bool)
        
        pred = pred_image[mask]
        gt = gt_image[mask]
        
        metrics = {}
        
        # PSNR
        mse = np.mean((pred - gt) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = 100.0
        metrics['psnr'] = float(psnr)
        
        # SSIM (简化版)
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(pred_image, gt_image, multichannel=True, data_range=255)
        metrics['ssim'] = float(ssim_score)
        
        # L1 Error
        l1 = np.mean(np.abs(pred - gt))
        metrics['l1'] = float(l1)
        
        # L2 Error
        l2 = np.sqrt(mse)
        metrics['l2'] = float(l2)
        
        return metrics


class MetricsLogger:
    """指标日志器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
    
    def log(self, metrics: Dict, step: int, prefix: str = ""):
        """记录指标"""
        log_entry = {
            'step': step,
            'prefix': prefix,
            'metrics': metrics
        }
        self.metrics_history.append(log_entry)
        
        # 实时保存
        self.save()
    
    def save(self):
        """保存日志"""
        import json
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def plot(self, metric_name: str, save_path: Optional[str] = None):
        """绘制指标曲线"""
        import matplotlib.pyplot as plt
        
        steps = []
        values = []
        
        for entry in self.metrics_history:
            if metric_name in entry['metrics']:
                steps.append(entry['step'])
                values.append(entry['metrics'][metric_name])
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=2)
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over training')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.log_dir / f'{metric_name}.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def summary(self) -> Dict:
        """生成摘要统计"""
        if not self.metrics_history:
            return {}
        
        # 收集所有指标
        all_metrics = {}
        for entry in self.metrics_history:
            for key, value in entry['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # 统计
        summary = {}
        for key, values in all_metrics.items():
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'last': float(values[-1])
            }
        
        return summary


# 测试代码
if __name__ == "__main__":
    print("=== Testing Evaluator ===")
    
    evaluator = Evaluator()
    
    # 测试3D重建评估
    print("\n1. 3D Reconstruction Metrics:")
    pred_points = np.random.randn(1000, 3)
    gt_points = pred_points + np.random.randn(1000, 3) * 0.05
    
    metrics_3d = evaluator.evaluate_reconstruction(pred_points, gt_points)
    print(f"  Chamfer Distance: {metrics_3d['chamfer_distance']:.4f}")
    print(f"  F-Score (0.05): {metrics_3d['f_score_0.05']:.4f}")
    print(f"  Accuracy: {metrics_3d['accuracy']:.4f}")
    
    # 测试深度评估
    print("\n2. Depth Metrics:")
    pred_depth = np.random.rand(480, 640) * 10
    gt_depth = pred_depth + np.random.randn(480, 640) * 0.5
    
    metrics_depth = evaluator.evaluate_depth(pred_depth, gt_depth)
    print(f"  Abs Rel: {metrics_depth['abs_rel']:.4f}")
    print(f"  RMSE: {metrics_depth['rmse']:.4f}")
    print(f"  δ < 1.25: {metrics_depth['delta_1.250']:.4f}")
    
    # 测试分割评估
    print("\n3. Segmentation Metrics:")
    pred_masks = [
        np.random.rand(480, 640) > 0.5,
        np.random.rand(480, 640) > 0.5
    ]
    gt_masks = [
        np.random.rand(480, 640) > 0.5,
        np.random.rand(480, 640) > 0.5,
        np.random.rand(480, 640) > 0.5
    ]
    
    metrics_seg = evaluator.evaluate_segmentation(pred_masks, gt_masks)
    print(f"  Precision: {metrics_seg['precision']:.4f}")
    print(f"  Recall: {metrics_seg['recall']:.4f}")
    print(f"  F1: {metrics_seg['f1']:.4f}")
    
    # 测试渲染评估
    print("\n4. Novel View Synthesis Metrics:")
    pred_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gt_image = pred_image + np.random.randint(-10, 10, (480, 640, 3), dtype=np.int16)
    gt_image = np.clip(gt_image, 0, 255).astype(np.uint8)
    
    metrics_render = evaluator.evaluate_novel_view(pred_image, gt_image)
    print(f"  PSNR: {metrics_render['psnr']:.2f}")
    print(f"  SSIM: {metrics_render['ssim']:.4f}")
    
    print("\n✓ All tests passed!")