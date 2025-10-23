from config import SEQ_LEN
from data.config import NUM_CONSORTIA_DAYS, OVERLAP
from data.normalization_functions.general import normalize_data_general
from data.normalization_functions.helena import normalize_data_helena
from data.normalization_functions.microbiome_gaussian import normalize_data_microbiome
from data.normalization_functions.rohan import normalize_data_rohan


def main():
    print("WARNING!! The microbiome code takes a while to run. Fitting all of the Gaussian processes is slow.")
    for name in [
        "fujita_microbiome",
        "huge_carin",
        "feilunconsortia",
        "schluter",
    ]:
        print(f"Normalizing {name}")
        normalize_data_microbiome(
            name=name,
            num_days=NUM_CONSORTIA_DAYS,
            length_input=SEQ_LEN,
            overlap=OVERLAP,
        )

    for name in [
        "alex",
        "cesar",
        "dongheon",
        "dan",    
        "emrah",
        "grayson",
        "jia",
        "katie",
        "kyeri",
        "lynch_lab",
        "zach",
        "zach_other",
        "zhixiang",
        "chory_lab", 
        "lopatkin_lab",
        "yuanchi",
        "shangying",
        "feilun",
        "carolyn",
        "sizhe",
        "harris",
        "ashwini",
        "hyein",
    ]:
        print(f"Normalizing {name}")
        normalize_data_general(
            name=name,
        )

    print(f"Normalizing helena")
    normalize_data_helena()

    print(f"Normalizing rohan")
    normalize_data_rohan()
