from applications.antibiotics.carolyn.main import main as classify_resistance
from applications.antibiotics.kyeri.classify_antibiotic_treatment import main as classify_treatment
from applications.antibiotics.kyeri.regress_antibiotic_concentration import main as regress_concentration


def main(
    model_type,
    z_dim,
    cross_val,
):
    print("Classifying treatment")
    classify_treatment(
        model_type=model_type,
        z_dim=z_dim,
        cross_val=cross_val,
    )
    print("Regressing concentration")
    regress_concentration(
        model_type=model_type,
        z_dim=z_dim,
        cross_val=cross_val,
    )
    print("Classifying resistance")
    classify_resistance(
        model_type=model_type,
        z_dim=z_dim,
        cross_val=cross_val,
    )
