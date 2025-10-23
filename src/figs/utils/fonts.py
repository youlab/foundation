import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from config import DIR_FIGS_FONT


def initialize_fonts():
    for _, _, files in os.walk(DIR_FIGS_FONT):
        break

    for file_name in files:
        fm.fontManager.addfont(DIR_FIGS_FONT / file_name)

    plt.rcParams['font.family'] = 'Open Sans'
