import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
import os
from typing import List, Dict, Any

warnings.filterwarnings('ignore')


class Processor:
    """
    A comprehensive processor for analyzing dashcam detection results with vibrant visualizations.
    
    This class provides methods to load, process, and visualize detection confidence scores
    and class performance metrics from dashcam analysis results.
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize the processor with detection results.
        
        Args:
            data (List[Dict[str, Any]]): List of detection results.
        """
        self.data = data
        
        # Set up modern styling
        plt.style.use('dark_background')
        sns.set_palette("bright")
        
        # Custom color palettes
        self.VIBRANT_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                               '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                               '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2']
        self.GRADIENT_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
        
        # Create custom colormap
        self.custom_cmap = LinearSegmentedColormap.from_list("vibrant", 
            ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"])
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Load and process the test results data."""
        if self.data is None:
            raise ValueError("No data provided to process.")
        
        all_scores = []
        all_labels = []
        test_case_ids = []
        
        for i, test_case in enumerate(self.data):
            all_scores.extend(test_case['scores'])
            all_labels.extend(test_case['labels'])
            test_case_ids.extend([i] * len(test_case['scores']))
        
        self.df = pd.DataFrame({
            'score': all_scores,
            'label': all_labels,
            'test_case': test_case_ids
        })

    def _plot_confidence_distribution(self):
        """Plot distribution of confidence scores with vibrant colors."""
        df = self.df
        if df is None:
            raise ValueError("No data available.")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Histogram with gradient colors
        n, bins, patches = ax1.hist(df['score'], bins=30, alpha=0.8, edgecolor='white', linewidth=1.5)
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.plasma(i / len(patches)))
        
        mean_val = df['score'].mean()
        median_val = df['score'].median()
        ax1.axvline(mean_val, color='#FF6B6B', linestyle='--', linewidth=3, 
                    label=f'Mean: {mean_val:.3f}', alpha=0.9)
        ax1.axvline(median_val, color='#4ECDC4', linestyle='--', linewidth=3, 
                    label=f'Median: {median_val:.3f}', alpha=0.9)
        
        ax1.set_xlabel('Confidence Score', fontsize=12, color='white', fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, color='white', fontweight='bold')
        ax1.set_title('Distribution of Confidence Scores', fontsize=14, color='white', fontweight='bold', pad=20)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, color='white')
        ax1.set_facecolor('#2d2d2d')
        
        ax2.boxplot(df['score'], vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='#45B7D1', alpha=0.8),
                    medianprops=dict(color='#FF6B6B', linewidth=3),
                    whiskerprops=dict(color='white', linewidth=2),
                    capprops=dict(color='white', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='#FFEAA7', markersize=8, alpha=0.7))
        
        ax2.set_ylabel('Confidence Score', fontsize=12, color='white', fontweight='bold')
        ax2.set_title('Confidence Score Distribution', fontsize=14, color='white', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, color='white')
        ax2.set_facecolor('#2d2d2d')
        
        plt.tight_layout()
        return fig

    def _plot_class_performance(self):
        """Plot performance by class with vibrant colors."""
        df = self.df
        if df is None:
            raise ValueError("No data available.")
            
        class_stats = df.groupby('label')['score'].agg(['mean', 'std', 'count']).reset_index()
        class_stats = class_stats.sort_values('mean', ascending=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
        fig.patch.set_facecolor('#1a1a1a')
        
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(class_stats)))
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(class_stats)))
        
        bars1 = ax1.barh(class_stats['label'], class_stats['mean'], 
                         xerr=class_stats['std'], capsize=4, color=colors1,
                         edgecolor='white', linewidth=1.5, alpha=0.9)
        ax1.set_xlabel('Mean Confidence Score', fontsize=12, color='white', fontweight='bold')
        ax1.set_title('Average Confidence Score by Class', fontsize=16, color='white', fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, color='white')
        ax1.set_facecolor('#2d2d2d')
        for i, (bar, mean_val) in enumerate(zip(bars1, class_stats['mean'])):
            ax1.text(mean_val + 0.005, bar.get_y() + bar.get_height()/2, 
                     f'{mean_val:.3f}', va='center', fontsize=10, 
                     color='white', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        bars2 = ax2.barh(class_stats['label'], class_stats['count'], 
                         color=colors2, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax2.set_xlabel('Number of Detections', fontsize=12, color='white', fontweight='bold')
        ax2.set_title('Detection Frequency by Class', fontsize=16, color='white', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, color='white')
        ax2.set_facecolor('#2d2d2d')
        for bar, count in zip(bars2, class_stats['count']):
            ax2.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                     f'{int(count)}', va='center', fontsize=10, 
                     color='white', fontweight='bold')
        
        plt.tight_layout()
        return fig

    def _plot_class_distribution(self):
        """Plot class distribution with vibrant colors."""
        df = self.df
        if df is None:
            raise ValueError("No data available.")
            
        class_counts = df['label'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        fig.patch.set_facecolor('#1a1a1a')
        
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        wedges, texts, autotexts = ax1.pie(class_counts.values, labels=class_counts.index, 
                                           autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                           explode=[0.05] * len(class_counts), shadow=True,
                                           textprops={'fontsize': 10, 'color': 'white'})
        ax1.set_title('Class Distribution (Pie Chart)', fontsize=16, color='white', fontweight='bold', pad=20)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        colors_bar = plt.cm.rainbow(np.linspace(0, 1, len(class_counts)))
        bars = ax2.barh(range(len(class_counts)), class_counts.values, 
                        color=colors_bar, edgecolor='white', linewidth=2, alpha=0.9)
        ax2.set_yticks(range(len(class_counts)))
        ax2.set_yticklabels(class_counts.index, color='white', fontweight='bold')
        ax2.set_xlabel('Count', fontsize=12, color='white', fontweight='bold')
        ax2.set_title('Class Distribution (Bar Chart)', fontsize=16, color='white', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, color='white', axis='x')
        ax2.set_facecolor('#2d2d2d')
        for bar, count in zip(bars, class_counts.values):
            ax2.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                     f'{int(count)}', va='center', fontsize=11, 
                     color='white', fontweight='bold')
        
        plt.tight_layout()
        return fig

    def _plot_confidence_by_class_detailed(self):
        """Detailed confidence analysis by class with vibrant colors."""
        df = self.df
        if df is None:
            raise ValueError("No data available.")
            
        fig, ax = plt.subplots(figsize=(18, 10))
        fig.patch.set_facecolor('#1a1a1a')
        
        class_order = df.groupby('label')['score'].mean().sort_values().index
        box_plot = sns.boxplot(data=df, x='label', y='score', order=class_order, ax=ax, 
                                palette='viridis', linewidth=2)
        for patch in box_plot.artists:
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(2)
        
        plt.xticks(rotation=45, ha='right', color='white', fontweight='bold', fontsize=11)
        ax.set_title('Confidence Score Distribution by Class', fontsize=18, color='white', fontweight='bold', pad=25)
        ax.set_ylabel('Confidence Score', fontsize=14, color='white', fontweight='bold')
        ax.set_xlabel('Class', fontsize=14, color='white', fontweight='bold')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('#2d2d2d')
        
        overall_mean = df['score'].mean()
        ax.axhline(overall_mean, color='#FF6B6B', linestyle='--', linewidth=3,
                   label=f'Overall Mean: {overall_mean:.3f}', alpha=0.9)
        ax.legend(fontsize=12, loc='upper left')
        
        plt.tight_layout()
        return fig

    def _plot_rainbow_performance_summary(self):
        """Create a comprehensive performance summary."""
        df = self.df
        if df is None:
            raise ValueError("No data available. Please run _load_and_process_data() first.")
            
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#1a1a1a')
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        n, bins, patches = ax1.hist(df['score'], bins=20, alpha=0.8, edgecolor='white')
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.rainbow(i / len(patches)))
        ax1.set_title('Score Distribution', color='white', fontweight='bold')
        ax1.set_facecolor('#2d2d2d')
        ax1.grid(True, alpha=0.3, color='white')
        
        ax2 = fig.add_subplot(gs[0, 1])
        top_classes = df.groupby('label')['score'].mean().sort_values(ascending=False).head(8)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_classes)))
        ax2.bar(range(len(top_classes)), top_classes.values, color=colors, edgecolor='white', linewidth=2)
        ax2.set_xticks(range(len(top_classes)))
        ax2.set_xticklabels(top_classes.index, rotation=45, ha='right', color='white')
        ax2.set_title('Top Performing Classes', color='white', fontweight='bold')
        ax2.set_facecolor('#2d2d2d')
        ax2.grid(True, alpha=0.3, color='white', axis='y')
        
        ax3 = fig.add_subplot(gs[1, 0])
        thresholds = [0.85, 0.90, 0.95, 0.98]
        threshold_counts = [(df['score'] > t).sum() for t in thresholds]
        colors = ['#74B9FF', '#FFEAA7', '#FF6B6B', '#A29BFE']
        ax3.bar([f'{t:.0%}' for t in thresholds], threshold_counts, color=colors, edgecolor='white', linewidth=2)
        ax3.set_title('Threshold Analysis', color='white', fontweight='bold')
        ax3.set_facecolor('#2d2d2d')
        ax3.grid(True, alpha=0.3, color='white', axis='y')
        
        ax4 = fig.add_subplot(gs[1, 1])
        class_counts = df['label'].value_counts().head(10)
        colors = plt.cm.plasma(np.linspace(0, 1, len(class_counts)))
        ax4.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90, textprops={'fontsize': 8, 'color': 'white'})
        ax4.set_title('Class Frequency', color='white', fontweight='bold')
        
        plt.suptitle('COMPREHENSIVE PERFORMANCE DASHBOARD', 
                     fontsize=20, color='white', fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig

    def _generate_summary_stats(self):
        """Generate and return summary statistics as a dict."""
        df = self.df
        if df is None:
            raise ValueError("No data available. Please run _load_and_process_data() first.")
            
        summary = {}

        summary['overall'] = {
            "total_detections": int(len(df)),
            "mean_confidence": float(df['score'].mean()),
            "median_confidence": float(df['score'].median()),
            "std_deviation": float(df['score'].std()),
            "min_score": float(df['score'].min()),
            "max_score": float(df['score'].max())
        }

        class_stats = df.groupby('label')['score'].agg(['count', 'mean', 'std']).round(4)
        class_stats = class_stats.sort_values('mean', ascending=False)
        # Convert class_stats to ensure Python native types
        class_performance_rankings = {}
        for index, row in class_stats.head(10).iterrows():
            std_val = row['std']
            class_performance_rankings[index] = {
                'count': int(row['count']),
                'mean': float(row['mean']),
                'std': float(std_val) if pd.notna(std_val) else None
            }
        summary['class_performance_rankings'] = class_performance_rankings

        thresholds = [0.98, 0.95, 0.90, 0.85]
        threshold_stats = {}
        for threshold in thresholds:
            count = int((df['score'] > threshold).sum())
            percentage = float((df['score'] > threshold).mean() * 100)
            threshold_stats[f'>{threshold:.0%}'] = {"count": count, "percentage": percentage}
        summary['confidence_thresholds'] = threshold_stats

        top_classes = df.groupby('label')['score'].mean().sort_values(ascending=False).head(5)
        summary['top_5_best_performing_classes'] = {label: float(score) for label, score in top_classes.items()}

        bottom_classes = df.groupby('label')['score'].mean().sort_values(ascending=True).head(5)
        summary['bottom_5_performing_classes'] = {label: float(score) for label, score in bottom_classes.items()}

        return summary
    
    def generate_report(self, save_path: str):
        """
        Generate a comprehensive report of the analysis.

        Args:
            save_path (str): Path to save the report as a JSON file.
        """
        print(save_path)

        summary = self._generate_summary_stats()

        fig_confidence_distribution = self._plot_confidence_distribution()
        fig_class_performance = self._plot_class_performance()
        fig_class_distribution = self._plot_class_distribution()
        fig_confidence_detailed = self._plot_confidence_by_class_detailed()
        fig_rainbow_summary = self._plot_rainbow_performance_summary()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            plots = {
                "confidence_distribution": os.path.join(os.path.basename(save_path), "confidence_distribution.png"),
                "class_performance": os.path.join(os.path.basename(save_path), "class_performance.png"),
                "class_distribution": os.path.join(os.path.basename(save_path), "class_distribution.png"),
                "confidence_detailed": os.path.join(os.path.basename(save_path), "confidence_detailed.png"),
                "rainbow_summary": os.path.join(os.path.basename(save_path), "rainbow_summary.png")
            }
            
            fig_confidence_distribution.savefig(os.path.join(save_path, "confidence_distribution.png"))
            fig_class_performance.savefig(os.path.join(save_path, "class_performance.png"))
            fig_class_distribution.savefig(os.path.join(save_path, "class_distribution.png"))
            fig_confidence_detailed.savefig(os.path.join(save_path, "confidence_detailed.png"))
            fig_rainbow_summary.savefig(os.path.join(save_path, "rainbow_summary.png"))
        else:
            plots = {}
        
        return {"summary": summary, "plots": plots}
