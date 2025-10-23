import numpy as np
from matplotlib import colormaps as cm

from config import DIR_DATA_EXP_PROC


def plot_data(
    a,
    filename,
    idx,
    y_type,
    x_ticks,
    title,
    write_x_label,
    fs_labels,
    fs_ticks,
    fs_text,
    lw=3,
    alpha=0.8,
    colormap_name="Blues",
):
    y_raw = np.load(
        DIR_DATA_EXP_PROC
        / filename.replace(
            "y.npy",
            "y_raw.npy",
        ),
        allow_pickle=True,
    )
    y_plot = y_raw[idx[0]:idx[1]]

    colormap = cm.get_cmap(colormap_name)
    colors = colormap(np.linspace(0.3, 1, idx[1] - idx[0],))

    if y_type == "OD":
        a.set_ylabel("OD600", fontsize=fs_labels,)
    elif y_type == "fraction":
        a.set_ylabel("Fraction", fontsize=fs_labels,)
    
    a.set_ylim((0, y_plot.max() * 1.1,))
    a.set_yticks(
        ticks=[
            np.around(y_plot.max() * 0.5, 2 if y_type == "fraction" else 1),
            np.around(y_plot.max(), 2 if y_type == "fraction" else 1),
        ],
        labels=[
            np.around(y_plot.max() * 0.5, 2 if y_type == "fraction" else 1),
            np.around(y_plot.max(), 2 if y_type == "fraction" else 1),
        ],
        fontsize=fs_ticks,
    )
    
    for i, y_ in enumerate(y_plot):
        a.plot(
            y_,
            lw=lw,
            color=colors[i],
            alpha=alpha,
        )
    
    if write_x_label:
        a.set_xlabel(
            "Time",
            fontsize=fs_labels,
        )

    a.set_xticks(
        ticks=x_ticks,
        labels=x_ticks,
        fontsize=fs_ticks,
    )
    a.set_xticks(
        ticks=[int(x_ticks[-1] * i) for i in [1/3, 2/3,]],
        minor=True,
    )
    
    x_title = 0
    y_title = y_plot.max() * 1.1
    a.text(
        x=x_title,
        y=y_title,
        s=title,
        fontsize=fs_text,
        verticalalignment="top",
    )
    for spine in ["top", "right"]: a.spines[spine].set_visible(False)


configs = [
    {
        "filename": "zach_growth_curves_2023-03-27_3_y.npy",
        "title": "Plate reader",
        "x_ticks": (0, 180,),
        "y_type": "OD",
        "idx": (0, 10,),
    },
    {
        "filename": "chory_lab_Kinetics Experiments_2022-04-11 psp AR2 comparison_2024-03-26 Absorbances_reader_plate_1_y.npy",
        "title": "Continuous\nculture",
        "x_ticks": (0, 45,),
        "y_type": "OD",
        "idx": (0, 10,),
    },
    {
        "filename": "fujita_microbiome_microbiome_dataset_y.npy",
        "title": "Microbiome",
        "x_ticks": (0, 120,),
        "y_type": "fraction",
        "idx": (590, 600,),
    }
]


def plot_three_types(
    ax=None,
    vertical=True,
    fs_labels=16,
    fs_ticks=14,
    fs_text=16,
    colormap_name="Blues",
):
    for i, config in enumerate(configs):
        plot_data(
            a=ax[i],
            filename=config.get("filename"),
            idx=config.get("idx"),
            y_type=config.get("y_type"),
            x_ticks=config.get("x_ticks"),
            title=config.get("title"),
            write_x_label=((not vertical) & (i == 0)) or (vertical & (i == (len(ax) - 1))),
            fs_labels=fs_labels,
            fs_ticks=fs_ticks,
            fs_text=fs_text,
            colormap_name=colormap_name,
        )
