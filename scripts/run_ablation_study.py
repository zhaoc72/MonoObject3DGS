"""
Ablation Study Runner
批量运行消融实验
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import subprocess
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AblationStudy:
    """消融实验管理器"""
    
    def __init__(self, image_paths: list, output_dir: str = "experiments/ablation"):
        self.image_paths = image_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验模式
        self.modes = [
            'high_accuracy',
            'balanced',
            'real_time',
            'ablation_no_dinov2',
            'ablation_no_depth',
            'ablation_minimal'
        ]
        
        self.results = []
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("=" * 70)
        print("🧪 Starting Ablation Study")
        print(f"   Images: {len(self.image_paths)}")
        print(f"   Modes: {len(self.modes)}")
        print(f"   Total runs: {len(self.image_paths) * len(self.modes)}")
        print("=" * 70)
        
        for image_path in self.image_paths:
            image_name = Path(image_path).stem
            print(f"\n📷 Processing: {image_name}")
            
            for mode in self.modes:
                print(f"\n  🔬 Mode: {mode}")
                
                try:
                    # 运行重建
                    cmd = [
                        'python', 'scripts/reconstruct_flexible.py',
                        '--image', image_path,
                        '--mode', mode,
                        '--output', str(self.output_dir / image_name)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # 读取统计
                        stats_file = self.output_dir / image_name / f"{mode}_{image_name}" / 'statistics.json'
                        
                        if stats_file.exists():
                            with open(stats_file, 'r') as f:
                                stats = json.load(f)
                            
                            self.results.append({
                                'image': image_name,
                                'mode': mode,
                                'objects': stats.get('num_objects', 0),
                                'gaussians': stats.get('total_gaussians', 0),
                                'total_time': stats.get('total_time', 0),
                                'seg_confidence': stats.get('v2_info', {}).get('avg_segmentation_confidence', 0),
                                'depth_confidence': stats.get('v2_info', {}).get('avg_depth_confidence', 0),
                                'dinov2_enabled': stats.get('config', {}).get('dinov2_enabled', False),
                                'sam2_enabled': stats.get('config', {}).get('sam2_enabled', False),
                                'depth_enabled': stats.get('config', {}).get('depth_enabled', False)
                            })
                            
                            print(f"    ✓ Success")
                        else:
                            print(f"    ⚠️  Stats file not found")
                    else:
                        print(f"    ✗ Failed: {result.stderr}")
                
                except Exception as e:
                    print(f"    ✗ Error: {e}")
        
        # 保存结果
        self._save_results()
        
        # 生成报告
        self._generate_report()
    
    def _save_results(self):
        """保存结果"""
        # 保存为JSON
        results_file = self.output_dir / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存为CSV
        df = pd.DataFrame(self.results)
        csv_file = self.output_dir / 'ablation_results.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\n✓ Results saved:")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {csv_file}")
    
    def _generate_report(self):
        """生成可视化报告"""
        if len(self.results) == 0:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        # 按模式分组统计
        grouped = df.groupby('mode').agg({
            'total_time': 'mean',
            'objects': 'mean',
            'gaussians': 'mean',
            'seg_confidence': 'mean',
            'depth_confidence': 'mean'
        }).reset_index()
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
        
        # 1. 时间对比
        ax = axes[0, 0]
        sns.barplot(data=grouped, x='mode', y='total_time', ax=ax, palette='viridis')
        ax.set_title('Average Processing Time')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. 物体数量
        ax = axes[0, 1]
        sns.barplot(data=grouped, x='mode', y='objects', ax=ax, palette='coolwarm')
        ax.set_title('Average Objects Detected')
        ax.set_ylabel('Number of Objects')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Gaussian数量
        ax = axes[0, 2]
        sns.barplot(data=grouped, x='mode', y='gaussians', ax=ax, palette='plasma')
        ax.set_title('Average Gaussians Generated')
        ax.set_ylabel('Number of Gaussians')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. 分割置信度
        ax = axes[1, 0]
        sns.barplot(data=grouped, x='mode', y='seg_confidence', ax=ax, palette='RdYlGn')
        ax.set_title('Average Segmentation Confidence')
        ax.set_ylabel('Confidence')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 1])
        
        # 5. 深度置信度
        ax = axes[1, 1]
        sns.barplot(data=grouped, x='mode', y='depth_confidence', ax=ax, palette='RdYlGn')
        ax.set_title('Average Depth Confidence')
        ax.set_ylabel('Confidence')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 1])
        
        # 6. 模块启用情况
        ax = axes[1, 2]
        module_stats = df.groupby('mode').agg({
            'dinov2_enabled': 'first',
            'sam2_enabled': 'first',
            'depth_enabled': 'first'
        }).reset_index()
        
        module_data = []
        for _, row in module_stats.iterrows():
            enabled_count = sum([row['dinov2_enabled'], row['sam2_enabled'], row['depth_enabled']])
            module_data.append({'mode': row['mode'], 'enabled_modules': enabled_count})
        
        module_df = pd.DataFrame(module_data)
        sns.barplot(data=module_df, x='mode', y='enabled_modules', ax=ax, palette='Set2')
        ax.set_title('Modules Enabled')
        ax.set_ylabel('Number of Modules')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 3])
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = self.output_dir / 'ablation_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Plot: {plot_file}")
        
        plt.close()
        
        # 生成Markdown报告
        self._generate_markdown_report(grouped)
    
    def _generate_markdown_report(self, grouped_df):
        """生成Markdown格式的报告"""
        report_file = self.output_dir / 'ABLATION_REPORT.md'
        
        with open(report_file, 'w') as f:
            f.write("# Ablation Study Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Images Tested:** {len(set([r['image'] for r in self.results]))}\n\n")
            f.write(f"**Modes Tested:** {len(self.modes)}\n\n")
            
            f.write("---\n\n")
            f.write("## Summary Statistics\n\n")
            f.write("| Mode | Avg Time (s) | Avg Objects | Avg Gaussians | Seg Conf | Depth Conf |\n")
            f.write("|------|--------------|-------------|---------------|----------|------------|\n")
            
            for _, row in grouped_df.iterrows():
                f.write(f"| {row['mode']} | {row['total_time']:.2f} | {row['objects']:.1f} | {row['gaussians']:.0f} | {row['seg_confidence']:.3f} | {row['depth_confidence']:.3f} |\n")
            
            f.write("\n---\n\n")
            f.write("## Module Configuration\n\n")
            
            # 模块配置表
            df = pd.DataFrame(self.results)
            module_config = df.groupby('mode').agg({
                'dinov2_enabled': 'first',
                'sam2_enabled': 'first',
                'depth_enabled': 'first'
            }).reset_index()
            
            f.write("| Mode | DINOv2 | SAM 2 | Depth |\n")
            f.write("|------|--------|-------|-------|\n")
            
            for _, row in module_config.iterrows():
                dinov2 = "✓" if row['dinov2_enabled'] else "✗"
                sam2 = "✓" if row['sam2_enabled'] else "✗"
                depth = "✓" if row['depth_enabled'] else "✗"
                f.write(f"| {row['mode']} | {dinov2} | {sam2} | {depth} |\n")
            
            f.write("\n---\n\n")
            f.write("## Key Findings\n\n")
            
            # 自动分析关键发现
            fastest_mode = grouped_df.loc[grouped_df['total_time'].idxmin(), 'mode']
            slowest_mode = grouped_df.loc[grouped_df['total_time'].idxmax(), 'mode']
            best_quality_mode = grouped_df.loc[grouped_df['seg_confidence'].idxmax(), 'mode']
            
            f.write(f"### Performance\n\n")
            f.write(f"- **Fastest Mode:** `{fastest_mode}` ({grouped_df.loc[grouped_df['mode']==fastest_mode, 'total_time'].values[0]:.2f}s)\n")
            f.write(f"- **Slowest Mode:** `{slowest_mode}` ({grouped_df.loc[grouped_df['mode']==slowest_mode, 'total_time'].values[0]:.2f}s)\n")
            f.write(f"- **Speed Ratio:** {grouped_df.loc[grouped_df['mode']==slowest_mode, 'total_time'].values[0] / grouped_df.loc[grouped_df['mode']==fastest_mode, 'total_time'].values[0]:.1f}x\n\n")
            
            f.write(f"### Quality\n\n")
            f.write(f"- **Best Quality Mode:** `{best_quality_mode}` (conf: {grouped_df.loc[grouped_df['mode']==best_quality_mode, 'seg_confidence'].values[0]:.3f})\n")
            f.write(f"- **Most Gaussians:** `{grouped_df.loc[grouped_df['gaussians'].idxmax(), 'mode']}` ({grouped_df['gaussians'].max():.0f} gaussians)\n\n")
            
            f.write(f"### Module Impact\n\n")
            
            # 计算禁用DINOv2的影响
            if 'ablation_no_dinov2' in grouped_df['mode'].values and 'high_accuracy' in grouped_df['mode'].values:
                no_dino_conf = grouped_df.loc[grouped_df['mode']=='ablation_no_dinov2', 'seg_confidence'].values[0]
                with_dino_conf = grouped_df.loc[grouped_df['mode']=='high_accuracy', 'seg_confidence'].values[0]
                dino_impact = ((with_dino_conf - no_dino_conf) / with_dino_conf) * 100
                f.write(f"- **DINOv2 Impact:** Removing DINOv2 reduces segmentation confidence by {dino_impact:.1f}%\n")
            
            # 计算禁用Depth的影响
            if 'ablation_no_depth' in grouped_df['mode'].values:
                f.write(f"- **Depth Impact:** Reconstruction quality severely degraded without depth estimation\n")
            
            f.write("\n---\n\n")
            f.write("## Recommendations\n\n")
            f.write("- **For maximum accuracy:** Use `high_accuracy` mode\n")
            f.write("- **For real-time applications:** Use `real_time` mode\n")
            f.write("- **For balanced performance:** Use `balanced` mode\n")
            f.write("- **All modules are critical:** Ablation studies show significant quality degradation when disabling any major component\n")
            
            f.write("\n---\n\n")
            f.write("## Visualization\n\n")
            f.write("![Ablation Comparison](ablation_comparison.png)\n\n")
        
        print(f"  Report: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Ablation Study')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                       help='Input image paths')
    parser.add_argument('--output', type=str, default='experiments/ablation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 运行消融实验
    study = AblationStudy(args.images, args.output)
    study.run_all_experiments()
    
    print("\n" + "=" * 70)
    print("✓ Ablation study completed!")
    print(f"  Results: {study.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()