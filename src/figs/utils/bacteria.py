import numpy as np
from matplotlib.patches import Wedge


def plot_living_bacteria(ax, x1, y1,):
    w = 0.05
    r1 = 0.2
    
    left_end = Wedge(
        center=(x1, y1),
        r=r1,
        theta1=134,
        theta2=316,
        width=0.25,
        color="g",
    )  # Ring sector
    ax.add_patch(left_end)
    
    
    x_match = x1 + (r1 - w) / np.sqrt(2)
    y_match = y1 - (r1 - w) / np.sqrt(2)
    
    r2 = 1.2
    
    x2 = x_match + r2 / np.sqrt(2)
    y2 = y_match - r2 / np.sqrt(2)
    back_edge = Wedge(
        center=(x2, y2,),
        r=r2,
        theta1=90,
        theta2=135,
        width=w,
        color="g",
    )
    ax.add_patch(back_edge)
    
    
    x_match = x2 * 1
    y_match = y2 + r2 - w
    
    r3 = r1
    x3 = x_match
    y3 = y_match + r3
    
    
    right_end = Wedge(
        center=(x3, y3),
        r=r3,
        theta1=269,
        theta2=91,
        width=0.25,
        color="g",
    )  # Ring sector
    ax.add_patch(right_end)
    
    r4 = r2 + r3 * 2 - w
    front_edge = Wedge(
        center=(x2, y2,),
        r=r4,
        theta1=90,
        theta2=135,
        width=r4-r2,
        color="g",
    )
    ax.add_patch(front_edge)


def plot_burst_bacteria(ax, x1, y1,):
    w = 0.05
    r1 = 0.2
    
    left_end = Wedge(
        center=(x1, y1),
        r=r1,
        theta1=134,
        theta2=316,
        width=w,
        color="k",
    )  # Ring sector
    ax.add_patch(left_end)
    
    
    x_match = x1 + (r1 - w) / np.sqrt(2)
    y_match = y1 - (r1 - w) / np.sqrt(2)
    
    r2 = 1.2
    
    x2 = x_match + r2 / np.sqrt(2)
    y2 = y_match - r2 / np.sqrt(2)
    back_edge = Wedge(
        center=(x2, y2,),
        r=r2,
        theta1=90,
        theta2=135,
        width=w,
        color="k",
    )
    ax.add_patch(back_edge)
    
    
    x_match = x2 * 1
    y_match = y2 + r2 - w
    
    r3 = r1
    x3 = x_match
    y3 = y_match + r3
    
    
    right_end = Wedge(
        center=(x3, y3),
        r=r3,
        theta1=269,
        theta2=91,
        width=w,
        color="k",
    )  # Ring sector
    ax.add_patch(right_end)
    
    r4 = r2 + r3 * 2 - w
    front_edge = Wedge(
        center=(x2, y2,),
        r=r4,
        theta1=90,
        theta2=105,
        width=w,
        color="k",
    )
    ax.add_patch(front_edge)

    front_edge = Wedge(
        center=(x2, y2,),
        r=r4,
        theta1=120,
        theta2=135,
        width=w,
        color="k",
    )
    ax.add_patch(front_edge)

    x5, y5 = x2 + np.cos(112.5 * np.pi * 2 / 360) * r4, y2 + np.sin(112.5/360 * np.pi * 2) * r4
    for i in range(6):
        burst = Wedge(
            center=(x5, y5,),
            r=0.4,
            theta1=60 + 20 * i,
            theta2=65 + 20 * i,
            width=0.2,
            color="k",
        )
        ax.add_patch(burst)


def plot_bacteria(ax):
    x1, y1 = 2.5, 0.3
    plot_burst_bacteria(ax, x1, y1,)
    x1, y1 = 0.8, 0.3
    plot_living_bacteria(ax, x1, y1,)

    ax.set_xlim(0, 4,)
    ax.set_ylim(0.07, 1.19,)
    
    ax.axis("off")
