import os
from pathlib import Path

home_dir = Path("/hpc/group/youlab/xw262")

DIR_RESULTS = home_dir / "foundation" / "results"
DIR_SRC = home_dir / "foundation" / "src"
DIR_DATA = DIR_SRC / "data"
DIR_MODELS = Path("/work/zah8/best_models/models")
DIR_LOGS = home_dir / "foundation" / "logs"

DIR_DATA_KARLSSON = DIR_SRC / "applications" / "karlsson" / "data"

DIR_DATA_CAROLYN = DIR_SRC / "applications" / "antibiotics" / "carolyn" / "data"
DIR_DATA_CHAOTIC = DIR_SRC / "data" / "simulation" / "chaotic"
DIR_DATA_EXP_PROC = DIR_SRC / "data" / "experimental" / "processed"
DIR_DATA_KYERI = DIR_SRC / "data" / "experimental" / "kyeri"
DIR_DATA_HELENA = DIR_SRC / "data" / "experimental" / "helena"
DIR_DATA_PROCESSED = DIR_SRC / "data" / "processed"
DIR_DATA_ZACH = DIR_SRC / "data" / "experimental" / "zach"
PATH_DATA_ZZ294_SIMPLE = DIR_SRC / "applications" / "consortia" / "data_files" / "bgLV_B15_T5_fixed.txt"
PATH_DATA_ZZ294_COMPLEX = DIR_SRC / "applications" / "consortia" / "data_files" / "dgLV_B92_T8_fixed.txt"
PATH_FUJITA_DATA = DIR_SRC / "data" / "experimental" / "fujita_microbiome" / "microbiome_dataset.csv"

DIR_CACHE_MODEL_FIGS = DIR_SRC / "figs" / "model" / "cache"
DIR_CACHE_CONSORTIA = DIR_SRC / "applications" / "consortia" / "cache"
DIR_CACHE_CONSORTIA_EXP = DIR_SRC / "applications" / "consortia_exp" / "cache"
DIR_CACHE_CAROLYN = DIR_SRC / "applications" / "antibiotics" / "carolyn" / "cache"
DIR_CACHE_HELENA = DIR_SRC / "applications" / "antibiotics" / "helena" / "cache"
DIR_CACHE_KYERI = DIR_SRC / "applications" / "antibiotics" / "kyeri" / "cache"

DIR_FIGS_MANUSCRIPT = DIR_RESULTS / "figs" / "manuscript"
DIR_FIGS_PRESENTATION = DIR_RESULTS / "figs" / "presentation"
DIR_FIGS_FONT = DIR_SRC / "figs" / "utils" / "open_sans"

DIR_RESULTS_DATA = DIR_RESULTS / "data"
DIR_RESULTS_MODEL = DIR_RESULTS / "model"
DIR_RESULTS_MODEL_NEW_SPLITS = DIR_RESULTS / "model_new_splits"
DIR_RESULTS_ANTIBIOTICS = DIR_RESULTS / "antibiotics"
DIR_RESULTS_CONSORTIA = DIR_RESULTS / "consortia"
DIR_RESULTS_CONSORTIA_EXP = DIR_RESULTS / "consortia_exp"
DIR_RESULTS_SUPER_RESOLUTION = DIR_RESULTS / "super_resolution"
DIR_RESULTS_MODEL_COMPARISON = DIR_RESULTS / "model_comparison"

SEQ_LEN = 128
Z_DIM = 8
MODEL_TYPE = "A7X"