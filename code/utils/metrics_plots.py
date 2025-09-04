"""
Professional metrics plotting module for UXO detection.

This module provides comprehensive plotting functionality for training metrics,
including MAP, AP, Recall evolution and Precision-Recall curves.
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, List, Optional
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_bar, geom_text, 
    geom_ribbon, facet_wrap, labs, theme, element_text, scale_color_manual,
    element_line, element_rect, scale_fill_manual
)
from plotnine.themes import theme_bw
from sklearn.metrics import precision_recall_curve, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning, module="plotnine")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotnine")


class MetricsPlotter:
    """
    Professional metrics plotting class for comprehensive model evaluation.
    
    This class provides methods to create publication-quality plots for:
    - Training metrics evolution (MAP, AP, Recall)
    - Precision-Recall curves
    - COCO metrics breakdown
    - Performance comparison plots
    """
    
    def __init__(self, plots_dir: str, logger=None):
        """
        Initialize the metrics plotter.
        
        Args:
            plots_dir: Directory to save plots
            logger: Logger instance for output
        """
        self.plots_dir = plots_dir
        self.logger = logger
        os.makedirs(plots_dir, exist_ok=True)
        
        self.colors = {
            'mAP': '#2E86AB',
            'mAP_50': '#A23B72',
            'mAP_75': '#F18F01',
            'mAR': '#C73E1D',
            'mAR_100': '#6B5B95',
            'Training': '#2E86AB',
            'Validation': '#A23B72',
            'Precision': '#F18F01',
            'Recall': '#C73E1D'
        }
    
    def plot_training_metrics_evolution(self, history: List[Dict]) -> None:
        """
        Create training metrics evolution plots.
        
        Args:
            history: List of training history dictionaries
        """
        df = pd.DataFrame(history)
        
        main_metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAR_100']
        available_metrics = [col for col in main_metrics if col in df.columns]
        
        if not available_metrics:
            self.logger.warning("No COCO metrics found in history for plotting")
            return
        
        main_df = df[['epoch'] + available_metrics].melt(
            id_vars=['epoch'], 
            var_name='metric', 
            value_name='value'
        )
        
        metric_mapping = {
            'mAP': 'mAP@0.5:0.95',
            'mAP_50': 'mAP@0.5',
            'mAP_75': 'mAP@0.75',
            'mAR_100': 'mAR@100'
        }
        main_df['metric'] = main_df['metric'].map(metric_mapping)
        
        main_plot = (
            ggplot(main_df, aes(x='epoch', y='value', color='metric'))
            + geom_line(size=1.2, alpha=0.8)
            + geom_point(size=2.5, alpha=0.7)
            + scale_color_manual(values=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            + labs(
                title="Model Performance Metrics Evolution",
                subtitle="Evaluation metrics over training epochs",
                x="Epoch",
                y="Metric Value",
                color="Metric"
            )
            + theme_bw()
            + theme(
                plot_title=element_text(size=16, weight="bold", hjust=0.5),
                plot_subtitle=element_text(size=12, hjust=0.5),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                legend_title=element_text(size=12, weight="bold"),
                legend_text=element_text(size=11),
                legend_position="top",
                panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                panel_border=element_rect(color="#B0B0B0", size=1)
            )
        )
        
        main_plot.save(f"{self.plots_dir}/metrics_evolution.png", dpi=300, width=14, height=10)
        
        size_metrics = ['mAP_small', 'mAP_medium', 'mAP_large']
        available_size_metrics = [col for col in size_metrics if col in df.columns]
        
        if available_size_metrics:
            size_df = df[['epoch'] + available_size_metrics].melt(
                id_vars=['epoch'], 
                var_name='metric', 
                value_name='value'
            )
            
            size_mapping = {
                'mAP_small': 'Small Objects',
                'mAP_medium': 'Medium Objects',
                'mAP_large': 'Large Objects'
            }
            size_df['metric'] = size_df['metric'].map(size_mapping)
            
            size_plot = (
                ggplot(size_df, aes(x='epoch', y='value', color='metric'))
                + geom_line(size=1.2, alpha=0.8)
                + geom_point(size=2.5, alpha=0.7)
                + scale_color_manual(values=['#2E86AB', '#A23B72', '#F18F01'])
                + labs(
                    title="mAP by Object Size",
                    subtitle="Performance across different object scales",
                    x="Epoch",
                    y="mAP Value",
                    color="Object Size"
                )
                + theme_bw()
                + theme(
                    plot_title=element_text(size=16, weight="bold", hjust=0.5),
                    plot_subtitle=element_text(size=12, hjust=0.5),
                    axis_title=element_text(size=13, weight="bold"),
                    axis_text=element_text(size=11),
                    legend_title=element_text(size=12, weight="bold"),
                    legend_text=element_text(size=11),
                    legend_position="top",
                    panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                    panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                    panel_border=element_rect(color="#B0B0B0", size=1)
                )
            )
            
            size_plot.save(f"{self.plots_dir}/object_size_metrics.png", dpi=300, width=14, height=10)
        
        if self.logger:
            self.logger.info(f"Training metrics evolution plots saved to {self.plots_dir}")
    
    def plot_precision_recall_curve(self, y_true: List, y_scores: List, 
                                   title: str = "Precision-Recall Curve") -> None:
        """
        Create professional Precision-Recall curve.
        
        Args:
            y_true: Ground truth labels
            y_scores: Predicted scores
            title: Plot title
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)
        
        pr_df = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'thresholds': np.append(thresholds, 1.0)
        })
        
        pr_plot = (
            ggplot(pr_df, aes(x='recall', y='precision'))
            + geom_line(color='#2E86AB', size=1.5, alpha=0.8)
            + geom_point(color='#2E86AB', size=2, alpha=0.7)
            + labs(
                title=title,
                subtitle=f"Average Precision: {ap_score:.4f}",
                x="Recall",
                y="Precision"
            )
            + theme_bw()
            + theme(
                plot_title=element_text(size=16, weight="bold", hjust=0.5),
                plot_subtitle=element_text(size=12, hjust=0.5),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                panel_border=element_rect(color="#B0B0B0", size=1)
            )
        )
        
        pr_plot.save(f"{self.plots_dir}/precision_recall_curve.png", dpi=300, width=12, height=8)
        
        if self.logger:
            self.logger.info(f"Precision-Recall curve saved to {self.plots_dir}")
    
    def plot_coco_metrics_breakdown(self, metrics: Dict[str, float]) -> None:
        """
        Create COCO metrics .
        
        Args:
            metrics: Dictionary of COCO metrics
        """
        # AP metrics
        ap_metrics = {
            'mAP@0.5:0.95': metrics.get('mAP', 0),
            'mAP@0.5': metrics.get('mAP_50', 0),
            'mAP@0.75': metrics.get('mAP_75', 0)
        }
        
        ap_df = pd.DataFrame([
            {'Metric': k, 'Value': v, 'Type': 'AP'}
            for k, v in ap_metrics.items()
        ])
        ap_df['Value_Formatted'] = ap_df['Value'].apply(lambda x: f'{x:.4f}')
        
        ap_plot = (
            ggplot(ap_df, aes(x='Metric', y='Value', fill='Metric'))
            + geom_bar(stat='identity', width=0.7, alpha=0.8)
            + geom_text(aes(label='Value_Formatted'), nudge_y=0.01, size=11, fontweight="bold")
            + labs(
                title="Average Precision Metrics",
                subtitle="Detection accuracy at different IoU thresholds",
                x="Metric",
                y="Value",
                fill="Metric"
            )
            + theme_bw()
            + theme(
                plot_title=element_text(size=16, weight="bold", hjust=0.5),
                plot_subtitle=element_text(size=12, hjust=0.5),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                legend_position="none",
                panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                panel_border=element_rect(color="#B0B0B0", size=1)
            )
        )
        
        ap_plot.save(f"{self.plots_dir}/ap_metrics.png", dpi=300, width=10, height=8)
        
        # AR metrics
        ar_metrics = {
            'AR@1': metrics.get('mAR_1', 0),
            'AR@10': metrics.get('mAR_10', 0),
            'AR@100': metrics.get('mAR_100', 0)
        }
        
        ar_df = pd.DataFrame([
            {'Metric': k, 'Value': v, 'Type': 'AR'}
            for k, v in ar_metrics.items()
        ])
        ar_df['Value_Formatted'] = ar_df['Value'].apply(lambda x: f'{x:.4f}')
        
        ar_plot = (
            ggplot(ar_df, aes(x='Metric', y='Value', fill='Metric'))
            + geom_bar(stat='identity', width=0.7, alpha=0.8)
            + geom_text(aes(label='Value_Formatted'), nudge_y=0.01, size=11, fontweight="bold")
            + labs(
                title="Average Recall Metrics",
                subtitle="Detection recall at different detection limits",
                x="Metric",
                y="Value",
                fill="Metric"
            )
            + theme_bw()
            + theme(
                plot_title=element_text(size=16, weight="bold", hjust=0.5),
                plot_subtitle=element_text(size=12, hjust=0.5),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                legend_position="none",
                panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                panel_border=element_rect(color="#B0B0B0", size=1)
            )
        )
        
        ar_plot.save(f"{self.plots_dir}/ar_metrics.png", dpi=300, width=10, height=8)
        
        if self.logger:
            self.logger.info(f"COCO metrics breakdown plots saved to {self.plots_dir}")
    
    def plot_performance_comparison(self, results: Dict[str, float]) -> None:
        """
        Create performance comparison plots.
        
        Args:
            results: Dictionary with performance results
        """
        # Performance metrics
        perf_metrics = {
            'FPS': results.get('fps', 0),
            'Avg Latency (s)': results.get('avg_time', 0),
            'mAP': results.get('mAP', 0),
            'AR@100': results.get('mAR_100', 0)
        }
        
        perf_df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in perf_metrics.items()
        ])
        perf_df['Value_Formatted'] = perf_df['Value'].apply(lambda x: f'{x:.2f}')
        
        perf_plot = (
            ggplot(perf_df, aes(x='Metric', y='Value', fill='Metric'))
            + geom_bar(stat='identity', width=0.7, alpha=0.8)
            + geom_text(aes(label='Value_Formatted'), nudge_y=perf_df['Value'].max() * 0.05, size=11, fontweight="bold")
            + labs(
                title="Model Performance Summary",
                subtitle="Perfomance Evaluation",
                x="Metric",
                y="Value",
                fill="Metric"
            )
            + theme_bw()
            + theme(
                plot_title=element_text(size=16, weight="bold", hjust=0.5),
                plot_subtitle=element_text(size=12, hjust=0.5),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                legend_position="none",
                panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                panel_border=element_rect(color="#B0B0B0", size=1)
            )
        )
        
        perf_plot.save(f"{self.plots_dir}/performance_summary.png", dpi=300, width=12, height=8)
        
        if self.logger:
            self.logger.info(f"Performance comparison plots saved to {self.plots_dir}")

    def plot_object_size_metrics_test(self, metrics: Dict[str, float]) -> None:
        """
        Create object size metrics plot for test results.
        
        Args:
            metrics: Dictionary with object size metrics
        """
        # Extract object size metrics
        size_metrics = {
            'Small Objects': metrics.get('mAP_small', 0),
            'Medium Objects': metrics.get('mAP_medium', 0),
            'Large Objects': metrics.get('mAP_large', 0)
        }
        
        # Create DataFrame
        size_df = pd.DataFrame([
            {'Object Size': k, 'mAP': v}
            for k, v in size_metrics.items()
        ])
        size_df['mAP_Formatted'] = size_df['mAP'].apply(lambda x: f'{x:.4f}')
        
       
        size_plot = (
            ggplot(size_df, aes(x='Object Size', y='mAP', fill='Object Size'))
            + geom_bar(stat='identity', width=0.7, alpha=0.8)
            + geom_text(aes(label='mAP_Formatted'), nudge_y=0.01, size=11, fontweight="bold")
            + scale_fill_manual(values=['#2E86AB', '#A23B72', '#F18F01'])
            + labs(
                title="mAP by Object Size - Test Results",
                subtitle="Detection performance across different object scales",
                x="Object Size",
                y="mAP Value",
                fill="Object Size"
            )
            + theme_bw()
            + theme(
                plot_title=element_text(size=16, weight="bold", hjust=0.5),
                plot_subtitle=element_text(size=12, hjust=0.5),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                legend_position="none",
                panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                panel_border=element_rect(color="#B0B0B0", size=1)
            )
        )
        
        size_plot.save(f"{self.plots_dir}/object_size_metrics.png", dpi=300, width=10, height=8)
        
        if self.logger:
            self.logger.info(f"Object size metrics test plot saved to {self.plots_dir}")
    
    def plot_comprehensive_test_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Create test metrics visualization.
        
        Args:
            metrics: Dictionary with test metrics
        """
        # Create a comprehensive metrics summary
        comprehensive_metrics = {
            'mAP@0.5:0.95': metrics.get('mAP', 0),
            'mAP@0.5': metrics.get('mAP_50', 0),
            'mAP@0.75': metrics.get('mAP_75', 0),
            'AR@100': metrics.get('mAR_100', 0),
            'Small Objects': metrics.get('mAP_small', 0),
            'Medium Objects': metrics.get('mAP_medium', 0),
            'Large Objects': metrics.get('mAP_large', 0)
        }
        
        comp_df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in comprehensive_metrics.items()
        ])
        comp_df['Value_Formatted'] = comp_df['Value'].apply(lambda x: f'{x:.4f}')
        
        # Create comprehensive bar plot
        comp_plot = (
            ggplot(comp_df, aes(x='Metric', y='Value', fill='Metric'))
            + geom_bar(stat='identity', width=0.7, alpha=0.8)
            + geom_text(aes(label='Value_Formatted'), nudge_y=0.01, size=10, fontweight="bold")
            + scale_fill_manual(values=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95', '#8B4513', '#228B22'])
            + labs(
                title="Test Metrics",
                subtitle="Evaluation results",
                x="Metric",
                y="Value",
                fill="Metric"
            )
            + theme_bw()
            + theme(
                plot_title=element_text(size=16, weight="bold", hjust=0.5),
                plot_subtitle=element_text(size=12, hjust=0.5),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=10, angle=45, hjust=1),
                legend_position="none",
                panel_grid_major=element_line(color="#E5E5E5", size=0.5),
                panel_grid_minor=element_line(color="#F2F2F2", size=0.3),
                panel_border=element_rect(color="#B0B0B0", size=1)
            )
        )
        
        comp_plot.save(f"{self.plots_dir}/comprehensive_test_metrics.png", dpi=300, width=12, height=8)
        
        if self.logger:
            self.logger.info(f"Test metrics plot saved to {self.plots_dir}")


def create_comprehensive_plots(history: List[Dict], metrics: Dict[str, float], 
                             plots_dir: str, logger=None) -> None:
    """
    Create all plots for model evaluation.
    
    Args:
        history: Training history
        metrics: Final evaluation metrics
        plots_dir: Directory to save plots
        logger: Logger instance
    """
    plotter = MetricsPlotter(plots_dir, logger)
    

    plotter.plot_training_metrics_evolution(history)
    

    plotter.plot_coco_metrics_breakdown(metrics)
    

    plotter.plot_performance_comparison(metrics)
    
    if logger:
        logger.info(f"All plots saved to {plots_dir}") 