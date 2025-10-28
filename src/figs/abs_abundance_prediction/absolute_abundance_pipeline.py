import numpy as np
import matplotlib.pyplot as plt

from figs.utils.symbols import (
    add_arrow,
    get_vae_patches,
    plot_rounded_rectangle,
)
from figs.utils.colors import get_colors
from matplotlib.patches import Rectangle


def add_mlp_rectangle(ax, xy, width, height):
    """Add a styled MLP rectangle (for regression, not classification)."""
    ax.add_patch(
        Rectangle(
            xy=xy,
            width=width,
            height=height,
            facecolor="gold",
            edgecolor="k",
            linewidth=5,  # Even thicker border for better visibility
            alpha=0.6,
        )
    )
    return xy[0] + width


def plot_absolute_abundance_pipeline(
    ax,
    fs_label=20,
    fs_text=16,
    show_example_data=True,
    show_latent_pathway=True,
    show_raw_pathway=True,
):
    """
    Visualize the pipeline for absolute abundance prediction from relative abundances.
    
    Shows two parallel pathways:
    1. Raw pathway: Relative abundances → MLP → Total abundance
    2. Latent pathway: Relative abundances → VAE Encoder → Latent + max → MLP → Total abundance
    
    Args:
        ax: Matplotlib axis
        fs_label: Font size for main labels
        fs_text: Font size for annotations
        show_example_data: Whether to show example trajectory curves
    """
    
    X_SPACE = 0.03  # Further reduced spacing
    ARROW_LEN = 0.15  # Shorter arrows
    
    # Define vertical positions for the pathways
    y_pos = 1.0  # Center the pathway vertically - ADJUSTED
    
    # --- INPUT: Relative abundance data ---
    # Different starting positions for panels A and B to avoid overlap
    if show_latent_pathway:
        left = -0.22  # Panel A: far left to utilize space
    else:  # show_raw_pathway (Panel B)
        left = 0.0  # Panel B: more to the right to avoid overlap with Panel A
    
    width_input = 0.75  # INCREASED width so curves look better (not shrunk)
    
    # Add pathway label CLOSER TO CURVES - ALIGNED AT SAME POSITION
    if show_latent_pathway:
        ax.text(
            -0.14,  # ADJUSTED for further left shift
            0.4,  # Same y-position for both labels
            "Using latents",  # Single line
            fontsize=fs_label - 2,  # Larger font
            ha="left",
            va="bottom",
            weight="bold",
            color="tab:blue",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
    if show_raw_pathway:
        ax.text(
            0.08,  # Panel B: positioned to the right
            0.4,  # Same y-position for both labels - ALIGNED
            "Using raw curves",  # Single line
            fontsize=fs_label - 2,  # Larger font
            ha="left",
            va="bottom",
            weight="bold",
            color="tab:orange",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
    
    # Add input label
    ax.text(
        left + width_input / 2,
        y_pos + 0.50,  # Above the curves - adjusted for new y_pos
        "Relative abundance\n(20 sp × 128 t)",
        fontsize=fs_label - 2,  # Larger font
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9, edgecolor="black", linewidth=2)
    )
    
    if show_example_data:
        # Plot example relative abundance trajectories
        n_species_to_show = 5
        colors = get_colors(n=n_species_to_show)
        t = np.linspace(0, 1, 50)
        
        # Generate synthetic-looking trajectories
        np.random.seed(42)
        for i, color in enumerate(colors):
            # Create varied dynamics
            freq = 0.5 + i * 0.3
            phase = i * 0.5
            amplitude = 0.12 + 0.08 * (i % 2)
            trajectory = 0.2 + amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Scale and position - ENLARGED and WELL-PROPORTIONED
            y_scaled = trajectory * 0.9 + y_pos - 0.28  # Larger vertical scale
            x_scaled = t * width_input + left
            
            if show_latent_pathway:
                ax.plot(x_scaled, y_scaled, lw=4.0, color=color, alpha=0.9)  # Even thicker lines
            if show_raw_pathway:
                ax.plot(x_scaled, y_scaled, lw=4.0, color=color, alpha=0.9)  # Even thicker lines
    
    left = left + width_input + X_SPACE
    
    if show_latent_pathway:
        # --- LATENT PATHWAY ---
        # Arrow to VAE encoder
        right = add_arrow(
            a=ax,
            x=left,
            y=y_pos,
            dx=ARROW_LEN,
            dy=0,
        )
        left = right + X_SPACE
        
        # VAE Encoder - ENLARGED
        h_vae = 0.9  # ENLARGED height
        w_vae = h_vae * 1.1
        right = get_vae_patches(
            a=ax,
            left=left,
            just_encoder_to_latent=True,
            bottom=y_pos - h_vae / 2,
            w_total=w_vae,
            h_total=h_vae,
        )
        
        left = right + X_SPACE
        
        # Arrow from encoder
        right = add_arrow(
            a=ax,
            x=left,
            y=y_pos,
            dx=ARROW_LEN * 0.7,
            dy=0,
        )
        left = right + X_SPACE
        
        # Add max value annotation - ENLARGED
        box_h = 0.65  # ENLARGED height
        box_w = 0.5  # ENLARGED width
        ax.add_patch(
            Rectangle(
                (left, y_pos - box_h/2),
                box_w,
                box_h,
                facecolor="lightyellow",
                edgecolor="black",
                linewidth=3,  # Thicker border
            )
        )
        ax.text(
            left + box_w / 2,
            y_pos,
            "+max",
            fontsize=fs_text - 4,  # Larger font
            ha="center",
            va="center",
            weight="bold",
        )
        
        left = left + box_w + X_SPACE
        
        # Arrow to MLP
        right = add_arrow(
            a=ax,
            x=left,
            y=y_pos,
            dx=ARROW_LEN,
            dy=0,
        )
        left_mlp = right + X_SPACE
    
    if show_raw_pathway:
        # --- RAW PATHWAY ---
        # Long arrow to MLP - thinner than default for panel B
        arrow_length = 2 * ARROW_LEN
        right = add_arrow(
            a=ax,
            x=left,
            y=y_pos,
            dx=arrow_length,
            dy=0,
            width=arrow_length / 12,  # Thinner width (default is dx/7)
        )
        left_mlp = right + X_SPACE
    
    # MLP regressor - ENLARGED
    h_mlp = 0.7  # ENLARGED height
    w_mlp = 0.48  # ENLARGED width
    right = add_mlp_rectangle(
        ax=ax,
        xy=(left_mlp, y_pos - h_mlp / 2),
        width=w_mlp,
        height=h_mlp,
    )
    ax.text(
        left_mlp + w_mlp / 2,
        y_pos,
        "MLP",
        fontsize=fs_label,  # Larger font
        ha="center",
        va="center",
        color="black",
        weight="bold",
    )
    
    left = right + X_SPACE
    
    # Arrow to output
    right = add_arrow(
        a=ax,
        x=left,
        y=y_pos,
        dx=ARROW_LEN,
        dy=0,
    )
    output_x = right + X_SPACE
    
    # --- OUTPUT: Total abundance prediction ---
    width_output = 0.5  # ENLARGED
    
    # Plot example output - ENLARGED
    t_out = np.linspace(0, 1, 50)
    y_output_example = 0.35 * np.sin(2 * np.pi * 0.8 * t_out) + y_pos
    x_output_scaled = t_out * width_output + output_x
    ax.plot(x_output_scaled, y_output_example, 'b-', lw=4, label="Predicted")  # Thicker line
    
    # Add output label
    ax.text(
        output_x + width_output / 2,
        y_pos + 0.5,  # Adjusted position
        "Total abundance\n(128 t)",
        fontsize=fs_label - 1,  # Larger font
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9, edgecolor="black", linewidth=2)
    )
    
    # Set axis limits and turn off - ADJUSTED based on panel
    if show_latent_pathway:
        ax.set_xlim(-0.25, output_x + width_output + 0.05)  # Panel A: Extended left
    else:  # show_raw_pathway (Panel B)
        ax.set_xlim(-0.05, output_x + width_output + 0.05)  # Panel B: Less left margin to avoid overlap
    ax.set_ylim(0.05, 1.9)  # Reduced white space above and below
    ax.axis("off")