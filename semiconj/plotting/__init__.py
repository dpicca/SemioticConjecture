"""
Generalized plotting module for SemioticConjecture output visualization.

This module provides a comprehensive set of plotting functions to automatically
generate visualizations for all CSV files in the 'out' directory (excluding cleaned_input.csv).
It replaces and generalizes the plotting functionality previously scattered across the codebase.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import csv

logger = logging.getLogger(__name__)


def safe_import_matplotlib():
    """Safely import matplotlib with informative error handling."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        return plt, patches, np
    except ImportError as e:
        raise ImportError("matplotlib and numpy are required for plotting. Install with: pip install matplotlib numpy") from e


def check_plotting_dependencies() -> bool:
    """Check if plotting dependencies are available without raising exceptions."""
    try:
        import matplotlib.pyplot
        import numpy
        return True
    except ImportError:
        return False


def load_csv_data(csv_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load CSV data and return headers and rows.
    
    Args:
        csv_path: Path to the CSV file.
        
    Returns:
        Tuple of (headers, rows) where rows is a list of dictionaries.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    
    # Convert numeric strings to floats where possible
    for row in rows:
        for key, value in row.items():
            if value and key != 'id' and key != 'context' and key != 'metric':
                try:
                    row[key] = float(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
    
    return headers, rows


def detect_plot_type(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    """Auto-detect the most appropriate plot type based on data structure.
    
    Args:
        headers: Column headers from the CSV.
        rows: Data rows as list of dictionaries.
        
    Returns:
        String indicating plot type: 'components_bar', 'line', 'context_bar', 
        'heatmap', 'scatter', 'histogram', 'metric_display'
    """
    if not rows:
        return 'empty'
    
    # Check for specific patterns
    if 'S_bin' in headers and 'k95' in headers:
        return 'line'  # Frontier data
    
    if 'context' in headers and len(headers) == 2:
        return 'context_bar'  # Omega/Rho data
    
    if 'metric' in headers and 'value' in headers:
        return 'metric_display'  # Analyses data
    
    if 'id' in headers and 'context' in headers:
        return 'heatmap'  # D_effective data
    
    # Check for components (multiple numeric columns with id)
    numeric_cols = [h for h in headers if h not in ['id', 'context', 'metric'] and 
                   any(isinstance(row.get(h), (int, float)) for row in rows)]
    
    if 'id' in headers and len(numeric_cols) >= 4:
        return 'components_bar'  # S_metrics or D_intr with multiple components
    
    if len(numeric_cols) >= 2:
        return 'scatter'  # Multiple numeric columns for correlation
    
    if len(numeric_cols) == 1:
        return 'histogram'  # Single numeric column
    
    return 'scatter'  # Default fallback


def create_components_bar_plot(headers: List[str], rows: List[Dict[str, Any]], 
                              output_path: Path, title: str = "Component Analysis") -> None:
    """Create grouped bar chart for component analysis (S_metrics, D_intr)."""
    plt, _, np = safe_import_matplotlib()
    
    ids = [row['id'] for row in rows]
    numeric_cols = [h for h in headers if h not in ['id', 'S', 'D_intr'] and 
                   any(isinstance(row.get(h), (int, float)) for row in rows)]
    
    if not numeric_cols:
        logger.warning(f"No numeric columns found for components bar plot: {output_path}")
        return
    
    # Create subplot for each component
    n_components = len(numeric_cols)
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 2 * n_components))
    if n_components == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(ids)))
    
    for i, col in enumerate(numeric_cols):
        values = [row.get(col, 0) for row in rows]
        axes[i].bar(ids, values, color=colors, alpha=0.7)
        axes[i].set_title(f'{col.title().replace("_", " ")}')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_line_plot(headers: List[str], rows: List[Dict[str, Any]], 
                    output_path: Path, title: str = "Line Plot") -> None:
    """Create line plot (typically for Frontier data)."""
    plt, _, _ = safe_import_matplotlib()
    
    # Assume first column is X, second is Y
    x_col, y_col = headers[0], headers[1]
    x_values = [row[x_col] for row in rows]
    y_values = [row[y_col] for row in rows]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_context_bar_plot(headers: List[str], rows: List[Dict[str, Any]], 
                           output_path: Path, title: str = "Context Comparison") -> None:
    """Create bar plot for context-based data (Omega, Rho)."""
    plt, _, _ = safe_import_matplotlib()
    
    contexts = [row['context'] for row in rows]
    value_col = headers[1]  # Second column should be the value
    values = [row[value_col] for row in rows]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(contexts, values, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Context')
    plt.ylabel(value_col.replace('_', ' ').title())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_heatmap(headers: List[str], rows: List[Dict[str, Any]], 
                   output_path: Path, title: str = "Heatmap") -> None:
    """Create heatmap for id x context data (D_effective)."""
    plt, _, np = safe_import_matplotlib()
    
    # Extract unique ids and contexts
    ids = sorted(set(row['id'] for row in rows))
    contexts = sorted(set(row['context'] for row in rows))
    value_col = [h for h in headers if h not in ['id', 'context']][0]
    
    # Create matrix
    matrix = np.zeros((len(ids), len(contexts)))
    for row in rows:
        i = ids.index(row['id'])
        j = contexts.index(row['context'])
        matrix[i, j] = row[value_col]
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label=value_col.replace('_', ' ').title())
    plt.xticks(range(len(contexts)), contexts, rotation=45)
    plt.yticks(range(len(ids)), ids)
    plt.xlabel('Context')
    plt.ylabel('ID')
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_metric_display(headers: List[str], rows: List[Dict[str, Any]], 
                         output_path: Path, title: str = "Statistical Metrics") -> None:
    """Create display for statistical metrics (Analyses data)."""
    plt, patches, np = safe_import_matplotlib()
    
    metrics = [row['metric'] for row in rows]
    values = [row['value'] for row in rows]
    
    # Check if we have correlation metrics that need special treatment
    correlation_metrics = ['kendall_tau', 'pearson', 'spearman', 'correlation']
    is_correlation = any(any(corr_type in metric.lower() for corr_type in correlation_metrics) for metric in metrics)
    
    if len(metrics) == 1 and is_correlation:
        # Special visualization for single correlation metric
        create_correlation_gauge(metrics[0], values[0], output_path, title, plt, patches, np)
    else:
        # Default bar chart for multiple metrics or non-correlation metrics
        create_standard_metric_display(metrics, values, output_path, title, plt, np)


def create_correlation_gauge(metric: str, value: float, output_path: Path, title: str, plt, patches, np) -> None:
    """Create a gauge-style display for correlation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left panel: Gauge visualization
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.3, 1.2)
    
    # Draw correlation scale background
    theta = np.linspace(0, np.pi, 100)
    x_arc = np.cos(theta)
    y_arc = np.sin(theta)
    ax1.plot(x_arc, y_arc, 'k-', linewidth=2, alpha=0.3)
    
    # Color zones for interpretation
    zones = [
        (-1.0, -0.7, '#d32f2f', 'Strong Negative'),
        (-0.7, -0.3, '#ff9800', 'Moderate Negative'),
        (-0.3, 0.3, '#ffc107', 'Weak/No Correlation'),
        (0.3, 0.7, '#4caf50', 'Moderate Positive'),
        (0.7, 1.0, '#2e7d32', 'Strong Positive')
    ]
    
    for start, end, color, label in zones:
        mask = (x_arc >= start) & (x_arc <= end)
        if np.any(mask):
            ax1.fill_between(x_arc[mask], 0, y_arc[mask], color=color, alpha=0.3)
    
    # Draw needle pointing to the actual value
    if -1 <= value <= 1:
        angle = np.arccos(value)  # value to angle conversion
        needle_x = [0, np.cos(angle)]
        needle_y = [0, np.sin(angle)]
        ax1.plot(needle_x, needle_y, 'r-', linewidth=4, marker='o', markersize=8, markerfacecolor='red')
    
    # Add scale labels
    scale_positions = [-1, -0.5, 0, 0.5, 1]
    for pos in scale_positions:
        if pos >= -1 and pos <= 1:
            angle = np.arccos(pos)
            x_pos = 1.1 * np.cos(angle)
            y_pos = 1.1 * np.sin(angle)
            ax1.text(x_pos, y_pos, f'{pos:.1f}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add value display
    ax1.text(0, -0.2, f'{value:.6f}', ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Correlation Strength', fontsize=14, fontweight='bold', pad=20)
    
    # Right panel: Interpretation text
    ax2.axis('off')
    
    # Determine interpretation
    abs_value = abs(value)
    direction = "positive" if value > 0 else "negative" if value < 0 else "no"
    
    if abs_value >= 0.7:
        strength = "strong"
        color = '#2e7d32' if value > 0 else '#d32f2f'
    elif abs_value >= 0.3:
        strength = "moderate"
        color = '#4caf50' if value > 0 else '#ff9800'
    else:
        strength = "weak"
        color = '#ffc107'
    
    interpretation_text = f"""
Metric: {metric.replace('_', ' ').title()}
Value: {value:.6f}

Interpretation:
• Direction: {direction.title()} correlation
• Strength: {strength.title()}
• Magnitude: {abs_value:.3f}

Meaning:
"""
    
    if abs_value < 0.1:
        interpretation_text += "• No meaningful relationship\n• Variables are independent"
    elif abs_value < 0.3:
        interpretation_text += "• Weak relationship\n• Limited predictive value"
    elif abs_value < 0.7:
        interpretation_text += "• Moderate relationship\n• Some predictive value"
    else:
        interpretation_text += "• Strong relationship\n• High predictive value"
    
    ax2.text(0.05, 0.95, interpretation_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.1))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_standard_metric_display(metrics: List[str], values: List[float], output_path: Path, title: str, plt, np) -> None:
    """Create standard bar chart display for multiple metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(metrics)), values, alpha=0.8)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(abs(v) for v in values)*0.05, 
               f'{value:.6f}', ha='center', va='bottom' if value >= 0 else 'top')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_scatter_plot(headers: List[str], rows: List[Dict[str, Any]], 
                       output_path: Path, title: str = "Scatter Plot") -> None:
    """Create scatter plot for correlation analysis."""
    plt, _, _ = safe_import_matplotlib()
    
    # Use first two numeric columns
    numeric_cols = [h for h in headers if h not in ['id', 'context', 'metric'] and 
                   any(isinstance(row.get(h), (int, float)) for row in rows)]
    
    if len(numeric_cols) < 2:
        logger.warning(f"Not enough numeric columns for scatter plot: {output_path}")
        return
    
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    x_values = [row[x_col] for row in rows]
    y_values = [row[y_col] for row in rows]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.7, s=60)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_histogram(headers: List[str], rows: List[Dict[str, Any]], 
                    output_path: Path, title: str = "Distribution") -> None:
    """Create histogram for single numeric column."""
    plt, _, _ = safe_import_matplotlib()
    
    numeric_cols = [h for h in headers if h not in ['id', 'context', 'metric'] and 
                   any(isinstance(row.get(h), (int, float)) for row in rows)]
    
    if not numeric_cols:
        logger.warning(f"No numeric columns found for histogram: {output_path}")
        return
    
    col = numeric_cols[0]
    values = [row[col] for row in rows if isinstance(row[col], (int, float))]
    
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=min(10, len(values)), alpha=0.7, edgecolor='black')
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_csv_file(csv_path: Path, output_dir: Path, custom_title: Optional[str] = None) -> None:
    """Generate appropriate plot for a single CSV file.
    
    Args:
        csv_path: Path to the input CSV file.
        output_dir: Directory to save the plot.
        custom_title: Optional custom title for the plot.
    """
    try:
        headers, rows = load_csv_data(csv_path)
        plot_type = detect_plot_type(headers, rows)
        
        # Generate output filename and title
        base_name = csv_path.stem
        output_path = output_dir / f"{base_name}_{plot_type}.png"
        title = custom_title or f"{base_name.replace('_', ' ').title()} - {plot_type.replace('_', ' ').title()}"
        
        logger.info(f"Creating {plot_type} plot for {base_name}: {output_path}")
        
        # Create appropriate plot
        if plot_type == 'components_bar':
            create_components_bar_plot(headers, rows, output_path, title)
        elif plot_type == 'line':
            create_line_plot(headers, rows, output_path, title)
        elif plot_type == 'context_bar':
            create_context_bar_plot(headers, rows, output_path, title)
        elif plot_type == 'heatmap':
            create_heatmap(headers, rows, output_path, title)
        elif plot_type == 'metric_display':
            create_metric_display(headers, rows, output_path, title)
        elif plot_type == 'scatter':
            create_scatter_plot(headers, rows, output_path, title)
        elif plot_type == 'histogram':
            create_histogram(headers, rows, output_path, title)
        else:
            logger.warning(f"Unknown plot type '{plot_type}' for {base_name}")
            
    except Exception as e:
        logger.error(f"Failed to create plot for {csv_path}: {e}")


def generate_all_plots(out_dir: Path, plot_dir: Optional[Path] = None) -> None:
    """Generate plots for all CSV files in the output directory (excluding cleaned_input.csv).
    
    Args:
        out_dir: Directory containing CSV files to plot.
        plot_dir: Directory to save plots (defaults to out_dir/plots).
    """
    # Check plotting dependencies first
    if not check_plotting_dependencies():
        logger.error("Cannot generate plots: matplotlib and numpy are required but not installed.")
        logger.error("Install plotting dependencies with: pip install matplotlib numpy")
        logger.error("Alternatively, install with: pip install -r requirements-plot.txt")
        return
    
    if plot_dir is None:
        plot_dir = out_dir / "plots"
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files except cleaned_input.csv
    csv_files = [f for f in out_dir.glob("*.csv") if f.name != "cleaned_input.csv"]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {out_dir} for plotting")
        return
    
    logger.info(f"Generating plots for {len(csv_files)} CSV files in {plot_dir}")
    
    success_count = 0
    for csv_file in csv_files:
        try:
            plot_csv_file(csv_file, plot_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to create plot for {csv_file.name}: {e}")
    
    if success_count == len(csv_files):
        logger.info(f"Plot generation complete. All {success_count} plots saved successfully to {plot_dir}")
    elif success_count > 0:
        logger.warning(f"Plot generation complete with errors. {success_count}/{len(csv_files)} plots saved to {plot_dir}")
    else:
        logger.error(f"Plot generation failed. No plots were created from {len(csv_files)} CSV files.")


# Legacy compatibility functions (replacements for semiconj.reporting functions)
def plot_frontier_legacy(output_path: Path, frontier_points: List[Tuple[float, float]]) -> None:
    """Legacy compatibility replacement for maybe_plot_frontier."""
    # Convert to CSV-like format and use standard plotting
    headers = ['S_bin', 'k95']
    rows = [{'S_bin': s, 'k95': k} for s, k in frontier_points]
    create_line_plot(headers, rows, output_path, "Semiotic Frontier k(H,C)")


def plot_correlation_legacy(output_path: Path, x_values: List[float], y_values: List[float], 
                           tau: float, x_label: str = "S values", y_label: str = "D_intr values") -> None:
    """Legacy compatibility replacement for maybe_plot_correlation."""
    plt, _, _ = safe_import_matplotlib()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.6, s=50)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Correlation Analysis: Kendall τ = {tau:.6f}')
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()