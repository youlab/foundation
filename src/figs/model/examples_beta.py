import numpy as np

from applications.utils.latents import reconstruct
from config import SEQ_LEN


def format_small_axis(a):
    a.set_ylim(0, 1.1)
    for spine in ["top", "right",]:
        a.spines[spine].set_visible(False)
    a.set_xticks(
        [0, 64, 128,],
        labels=[],
    )
    a.set_yticks(
        [0, 0.5, 1,],
        labels=[],
    )


def plot_examples_beta(
    ax,
    data,
    idx,
    model,
    p_y=0.02,
    p_x=0.01,
    lw=4,
    fs_text=16,
    fw_text="medium",
):
    ax.axis("off")
    
    a = []
    n_panels = 16
    n_cols = 8
    n_rows = n_panels // n_cols
    x_step = 1 / n_cols
    y_step = 1 / n_rows
    
    data_examples = [
        ("experimental", 'zach_growth_curves_2023-03-27_3_y.npy', 0,),
        ("experimental", 'alex_2023-06-17_OD600_alex_y.npy', 10,),
        ("experimental", 'emrah_RawTimeCourse_Ch49_y.npy', 0,),
        ("experimental", 'helena_od600_y.npy', 20,),
        ("experimental", 'fujita_microbiome_microbiome_dataset_y.npy', 95,),
        ("experimental", 'huge_carin_subjectA_saliva_y.npy', 95,),
        ("experimental", 'huge_carin_subjectA_gut_y.npy', 95,),
        ("experimental", 'feilun_synth_comm_data_y.npy', 1,),
        ("simulation", 'lingchong_eqn_y.npy', 1090,),
        ("simulation", 'lingchong_eqn_y.npy', 1500,),
        ("simulation", 'antibiotic_y.npy', 1090,),
        ("simulation", 'antibiotic_y.npy', 1500,),
        ("simulation", "chaotic_2024-06-03-v0_y.npy", 102,),
        ("simulation", "chaotic_2024-06-03-v1_y.npy", 102,),
        ("simulation", "chaotic_2024-06-03-v2_y.npy", 100,),
        ("simulation", "chaotic_2024-06-03-v3_y.npy", 102,),
    ]
    
    i_example = 0
    for i_col in range(n_cols):
        for i_row in range(n_rows):
            category, file_name, i_mask = data_examples[i_example]
            mask = (idx[category][file_name]["y_all_i"] <= data["test_idx"]) * (data["test_idx"] < (idx[category][file_name]["y_all_i"] + idx[category][file_name]["y.shape"][0]))
            y_true = data["y"][data["test_idx"][mask][i_mask]]
            
            a = ax.inset_axes(
                [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
            )

            reconstruction = reconstruct(
                model=model,
                x=y_true.reshape(1, 1, -1),
                batch_size=1,
                is_vae=True,
                is_pca=False,
            )[0, 0, :]
    
            a.plot(
                np.arange(SEQ_LEN),
                y_true,
                color="cornflowerblue",
                lw=lw*1.1,
                alpha=0.8,
                label="original",
            )
            a.plot(
                np.arange(SEQ_LEN),
                reconstruction,
                color="firebrick",
                lw=lw*0.9,
                alpha=0.8,
                label="reconstruction",
            )
            if (i_col == 0) and (i_row == 1):
                a.text(
                    0,
                    0.6,
                    "original",
                    color="cornflowerblue",
                    fontsize=fs_text,
                    fontweight=fw_text,
                    alpha=0.8,
                )
                a.text(
                    0,
                    0.2,
                    "reconstruction",
                    color="firebrick",
                    fontsize=fs_text,
                    fontweight=fw_text,
                    alpha=0.8,
                )
            if (i_col == 0) and (i_row == 0):
                a.set_xlabel("Time", fontsize=fs_text+4)
                a.set_ylabel("Cell density", fontsize=fs_text+4)
            format_small_axis(a=a)

            i_example += 1
    