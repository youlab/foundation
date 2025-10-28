import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def plot_simple_comparison(
    ax,
    fs_title=18,
    fs_label=14,
):
    """
    A simplified side-by-side comparison of the two approaches.
    
    Shows a compact visual comparison without detailed pipeline elements.
    
    Args:
        ax: Matplotlib axis
        fs_title: Font size for titles
        fs_label: Font size for labels
    """
    
    # Define positions
    left_x = 0.1
    right_x = 0.55
    width = 0.35
    
    # Colors
    color_input = "#E3F2FD"  # Light blue
    color_process = "#FFF3E0"  # Light orange
    color_output = "#E8F5E9"  # Light green
    
    # === LEFT PANEL: RAW APPROACH ===
    
    # Title
    ax.text(
        left_x + width/2, 0.95,
        "Raw Pathway",
        fontsize=fs_title,
        weight='bold',
        ha='center',
        va='top',
        color='darkred'
    )
    
    # Input
    box1 = FancyBboxPatch(
        (left_x, 0.65), width, 0.2,
        boxstyle="round,pad=0.01",
        facecolor=color_input,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box1)
    ax.text(
        left_x + width/2, 0.75,
        "Relative\nAbundance\n(2560D)",
        fontsize=fs_label,
        ha='center',
        va='center'
    )
    
    # Arrow
    arrow1 = FancyArrowPatch(
        (left_x + width/2, 0.65),
        (left_x + width/2, 0.50),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow1)
    
    # Process
    box2 = FancyBboxPatch(
        (left_x, 0.35), width, 0.15,
        boxstyle="round,pad=0.01",
        facecolor=color_process,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box2)
    ax.text(
        left_x + width/2, 0.425,
        "MLP\nRegressor",
        fontsize=fs_label,
        ha='center',
        va='center',
        weight='bold'
    )
    
    # Arrow
    arrow2 = FancyArrowPatch(
        (left_x + width/2, 0.35),
        (left_x + width/2, 0.20),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow2)
    
    # Output
    box3 = FancyBboxPatch(
        (left_x, 0.05), width, 0.15,
        boxstyle="round,pad=0.01",
        facecolor=color_output,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box3)
    ax.text(
        left_x + width/2, 0.125,
        "Total\nAbundance\n(128D)",
        fontsize=fs_label,
        ha='center',
        va='center'
    )
    
    # === RIGHT PANEL: LATENT APPROACH ===
    
    # Title
    ax.text(
        right_x + width/2, 0.95,
        "Latent Pathway",
        fontsize=fs_title,
        weight='bold',
        ha='center',
        va='top',
        color='darkblue'
    )
    
    # Input (same)
    box4 = FancyBboxPatch(
        (right_x, 0.65), width, 0.2,
        boxstyle="round,pad=0.01",
        facecolor=color_input,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box4)
    ax.text(
        right_x + width/2, 0.75,
        "Relative\nAbundance\n(2560D)",
        fontsize=fs_label,
        ha='center',
        va='center'
    )
    
    # Arrow
    arrow3 = FancyArrowPatch(
        (right_x + width/2, 0.65),
        (right_x + width/2, 0.58),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow3)
    
    # Foundation Model Encoder
    box5 = FancyBboxPatch(
        (right_x, 0.48), width, 0.10,
        boxstyle="round,pad=0.01",
        facecolor='#E1BEE7',  # Light purple
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box5)
    ax.text(
        right_x + width/2, 0.53,
        "Foundation Model\nEncoder",
        fontsize=fs_label - 1,
        ha='center',
        va='center',
        weight='bold'
    )
    
    # Arrow
    arrow4 = FancyArrowPatch(
        (right_x + width/2, 0.48),
        (right_x + width/2, 0.42),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow4)
    
    # Latent features
    box6 = FancyBboxPatch(
        (right_x + 0.02, 0.35), width - 0.04, 0.07,
        boxstyle="round,pad=0.005",
        facecolor='#FFF9C4',  # Light yellow
        edgecolor='gray',
        linewidth=1.5,
        linestyle='--'
    )
    ax.add_patch(box6)
    ax.text(
        right_x + width/2, 0.385,
        "8D Latent + max = 9D",
        fontsize=fs_label - 2,
        ha='center',
        va='center',
        style='italic'
    )
    
    # Arrow
    arrow5 = FancyArrowPatch(
        (right_x + width/2, 0.35),
        (right_x + width/2, 0.28),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow5)
    
    # MLP
    box7 = FancyBboxPatch(
        (right_x, 0.13), width, 0.15,
        boxstyle="round,pad=0.01",
        facecolor=color_process,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box7)
    ax.text(
        right_x + width/2, 0.205,
        "MLP\nRegressor",
        fontsize=fs_label,
        ha='center',
        va='center',
        weight='bold'
    )
    
    # Arrow
    arrow6 = FancyArrowPatch(
        (right_x + width/2, 0.13),
        (right_x + width/2, 0.06),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow6)
    
    # Output (same)
    box8 = FancyBboxPatch(
        (right_x, -0.09), width, 0.15,
        boxstyle="round,pad=0.01",
        facecolor=color_output,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box8)
    ax.text(
        right_x + width/2, -0.015,
        "Total\nAbundance\n(128D)",
        fontsize=fs_label,
        ha='center',
        va='center'
    )
    
    # Add annotation
    ax.text(
        0.5, -0.18,
        "Key difference: Latent pathway uses 9D compressed features vs. 2560D raw features",
        fontsize=fs_label - 1,
        ha='center',
        va='top',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, edgecolor='none')
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 1)
    ax.axis('off')


def plot_hypothesis_diagram(ax, fs_text=14):
    """
    A simple diagram illustrating the hypothesis about foundation models.
    
    Args:
        ax: Matplotlib axis
        fs_text: Font size for text
    """
    
    # Title
    ax.text(
        0.5, 0.95,
        "Foundation Model Hypothesis",
        fontsize=fs_text + 6,
        weight='bold',
        ha='center',
        va='top'
    )
    
    # Question box
    question_box = FancyBboxPatch(
        (0.1, 0.7), 0.8, 0.15,
        boxstyle="round,pad=0.02",
        facecolor='#E3F2FD',
        edgecolor='#1976D2',
        linewidth=2
    )
    ax.add_patch(question_box)
    ax.text(
        0.5, 0.775,
        "Can foundation models extract features from\nrelative abundances to predict total abundances?",
        fontsize=fs_text,
        ha='center',
        va='center',
        weight='bold'
    )
    
    # Approach boxes
    y_start = 0.55
    box_height = 0.18
    
    # Approach 1
    box1 = FancyBboxPatch(
        (0.05, y_start - box_height), 0.4, box_height,
        boxstyle="round,pad=0.01",
        facecolor='#FFEBEE',
        edgecolor='#C62828',
        linewidth=2
    )
    ax.add_patch(box1)
    ax.text(
        0.25, y_start - 0.02,
        "Baseline: Raw Features",
        fontsize=fs_text,
        ha='center',
        va='top',
        weight='bold',
        color='#C62828'
    )
    ax.text(
        0.25, y_start - 0.12,
        "• All time points × species\n• No compression",
        fontsize=fs_text - 2,
        ha='center',
        va='center',
        linespacing=1.5
    )
    
    # Approach 2
    box2 = FancyBboxPatch(
        (0.55, y_start - box_height), 0.4, box_height,
        boxstyle="round,pad=0.01",
        facecolor='#E8EAF6',
        edgecolor='#283593',
        linewidth=2
    )
    ax.add_patch(box2)
    ax.text(
        0.75, y_start - 0.02,
        "Foundation: Latent Features",
        fontsize=fs_text,
        ha='center',
        va='top',
        weight='bold',
        color='#283593'
    )
    ax.text(
        0.75, y_start - 0.12,
        "• Compressed by encoder\n• Captures dynamics",
        fontsize=fs_text - 2,
        ha='center',
        va='center',
        linespacing=1.5
    )
    
    # Expected outcome
    outcome_box = FancyBboxPatch(
        (0.1, 0.05), 0.8, 0.12,
        boxstyle="round,pad=0.02",
        facecolor='#F1F8E9',
        edgecolor='#558B2F',
        linewidth=2
    )
    ax.add_patch(outcome_box)
    ax.text(
        0.5, 0.11,
        "Expected: Latent features should enable accurate prediction\ndespite 280× dimensionality reduction",
        fontsize=fs_text,
        ha='center',
        va='center',
        style='italic'
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


if __name__ == "__main__":
    """Test the simplified visualizations"""
    from pathlib import Path
    from config import DIR_FIGS_PRESENTATION
    
    # Create output directory if it doesn't exist
    DIR_FIGS_PRESENTATION.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {DIR_FIGS_PRESENTATION}\n")
    
    # Test simple comparison
    print("Generating Simple Comparison figure...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_simple_comparison(ax, fs_title=18, fs_label=14)
    output_path = DIR_FIGS_PRESENTATION / "abs_abundance_simple_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved simple comparison to: {output_path}")
    output_path_pdf = DIR_FIGS_PRESENTATION / "abs_abundance_simple_comparison.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved simple comparison to: {output_path_pdf}")
    plt.close()
    
    # Test hypothesis diagram
    print("\nGenerating Hypothesis Diagram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_hypothesis_diagram(ax, fs_text=14)
    output_path = DIR_FIGS_PRESENTATION / "abs_abundance_hypothesis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved hypothesis diagram to: {output_path}")
    output_path_pdf = DIR_FIGS_PRESENTATION / "abs_abundance_hypothesis.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved hypothesis diagram to: {output_path_pdf}")
    plt.close()
    
    print("\n Both figures generated successfully!")

