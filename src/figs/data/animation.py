import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from config import DIR_FIGS_PRESENTATION
from figs.utils.colors import get_colors
from data.utils import get_data


def calc_layer(layer):
    n_total = 1
    for m in range(1, layer+1):
        s = 2 * m - 1
        n_total += s * 4 - 4
    s = 2 * layer - 1
    n = s * 4 - 4
    if n == 0:
        n = 1
    x, y = -layer+1, -layer+1
    coords = []
    for i in range(n):
        coords.append((x, y))
        if i < (s - 1):
            x += 1
        elif i < (2 * s - 2):
            y += 1
        elif i < (3 * s - 3):
            x -= 1
        else:
            y -= 1
    return n_total, n, coords


def main():
    import os
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    data, _ = get_data()
    y = data["y"]
    np.random.shuffle(y)
    x = np.linspace(0, 1, 128)
    n_colors = 25
    colors = get_colors(n=n_colors)

    # chunk_size = 16
    # frames = 128 * 275 // chunk_size

    chunk_size1 = 8
    count1 = 2
    chunk_size2 = 16
    count2 = 5
    chunk_size3 = 128
    count3 = 50
    count4 = 28
    wait4 = 10
    frames = 128 // chunk_size1 * count1
    # print(frames)
    # print(frames)
    frames += 128 // chunk_size2 * count2
    # print(frames)
    # frames += 268
    frames += 128 // chunk_size3 * count3
    # print(frames)
    frames += count4 * wait4

    shift = 1

    def update(frame):
        # layer = frame // (128 // chunk_size) + 1
        # t = frame % (128 // chunk_size) * chunk_size
        print("frame", frame)
        shift = 0.9
        if frame < (128 // chunk_size1 * count1):
            chunk_size = chunk_size1
            layer = frame // (128 // chunk_size) + 1
            t = frame % (128 // chunk_size) * chunk_size
            step = 1
            n_total, _, coords = calc_layer(layer=layer)
            alpha = 0.8
            lw = 2
        elif frame < ((128 // chunk_size1 * count1) + (128 // chunk_size2 * count2)):
            chunk_size = chunk_size2
            frame -= (128 // chunk_size1 * count1)
            layer = frame // (128 // chunk_size) + 1 + count1
            t = frame % (128 // chunk_size) * chunk_size
            step = 1
            n_total, _, coords = calc_layer(layer=layer)
            alpha = 0.8
            lw = 2
        elif frame < ((128 // chunk_size1 * count1) + (128 // chunk_size2 * count2) + (128 // chunk_size3 * count3)):
            frame -= (128 // chunk_size1 * count1) + (128 // chunk_size2 * count2)
            chunk_size = chunk_size3
            layer = frame + 1 + count1 + count2
            t = 0
            step = 8
            n_total, _, coords = calc_layer(layer=layer)
            alpha = 0.8
            lw = 2
        else:
            frame -= (128 // chunk_size1 * count1) + (128 // chunk_size2 * count2) + (128 // chunk_size3 * count3)
            if (frame % wait4) != 0:
                return
            frame = frame // wait4
            n_total = 12769 + frame * 12769
            # i2 = 12769 + (frame + 1) * 12769
            t = 0
            step = 8
            chunk_size = chunk_size3
            coords = []
            for i_row in range(-56, 57):
                for i_col in range(-56, 57):
                    coords.append((i_row, i_col))
            alpha = 0.3
            lw = 1
        print("n_total", n_total)
        for i, (x_adder, y_adder) in enumerate(coords):
            t2 = t + chunk_size + 1
            i_y = n_total + i
            color = colors[i_y % n_colors]
            ax.plot(
                x[t:t2][0::step] + x_adder * shift,
                y[i_y, t:t2][0::step] + y_adder * shift,
                color=color,
                lw=lw,
                alpha=alpha,
            )

    print("Making animation")
    fig, ax = plt.subplots(1, 1, figsize=(16, 9,),)
    ax.axis("off")
    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=1, blit=False)
    ani.save(
        filename=DIR_FIGS_PRESENTATION
        / f"temp_random_{task_id}.gif",
        writer="pillow",
    )
    print("Saved animation")
