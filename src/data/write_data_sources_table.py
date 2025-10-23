import os

from config import (
    DIR_DATA,
    DIR_RESULTS_DATA,
)
import pandas as pd


def apply_entire_folder_to_one(ref, d, s,):
    for _, _, files in os.walk(d):
        break

    directory = f"{d}".split(f"{str(DIR_DATA)}/experimental/")[1]
    ref[directory] = {}
    for file_name in files:
        ref[directory][file_name] = s

    return ref


def apply_entire_folder_unpublished(ref, d,):
    for _, _, files in os.walk(d):
        break

    directory = f"{d}".split(f"{str(DIR_DATA)}/experimental/")[1]
    ref[directory] = {}
    for file_name in files:
        ref[directory][file_name] = "this study"

    return ref


def apply_lopatkin(
    ref,
    d,
):
    """
    KPN is for the Ahmad paper, and DG is for the palomino paper. For the others, if you open them 
    up there should be a details tab that says the experiment ID which should match one I’ve 
    already given to you. If not let me know and I can track it down on my end. Thanks!
    """
    for _, _, files in os.walk(d):
        break
    
    directory = f"{d}".split(f"{str(DIR_DATA)}/experimental/")[1]
    ref[directory] = {}
    for file_name in files:
        if (file_name.find("_tet") > -1) or (file_name.find("KPN") > -1) or (file_name.find("timecourses_all") > -1):
            ref[directory][file_name] = ahmad
        elif (file_name.find("BW") > -1) or (file_name.find("KS") > -1) or (file_name.find("Day21") > -1) or (file_name.find("Day0") > -1):
            ref[directory][file_name] = aduru
        elif (file_name.find("exp5") > -1) or (file_name.find("_AP") > -1) or (file_name.find("5.") > -1) or (file_name[:2] == "DG") or (file_name.find("ajl_growthKinetics") > -1):
            ref[directory][file_name] = palomino
        elif (file_name.find("7.") > -1):
            ref[directory][file_name] = prensky
        else:
            ref[directory][file_name] = "unknown"
    return ref


def apply_lynch(
    ref,
    d,
):
    for _, _, files in os.walk(d):
        break
    
    directory = f"{d}".split(f"{str(DIR_DATA)}/experimental/")[1]
    ref[directory] = {}
    ref[directory] = {}
    for file_name in sorted(files):
        if (
            file_name.find("PID") > -1
        ):
            ref[directory][file_name] = li_2021
        
        elif (file_name.find("_Manual") > -1
        ) or (
            file_name.find("IRIS") > -1
        ):
            ref[directory][file_name] = li_2024
        elif (file_name.find("Process") > -1) or (file_name.find("DMC_Ferm") > -1):
            ref[directory][file_name] = ye
        elif file_name.find("autoinduction_best_screened media_mcherry_") > -1:
            ref[directory][file_name] = menacho
        elif file_name.find("Promoters_BioLector_OD_Corr_Promoter_Data") > -1:
            ref[directory][file_name] = moreb
        elif (file_name.find("Biolector") > -1) or (file_name.find("VHH_VCAM") > -1) or (file_name.find("nanobodies") > -1):
            ref[directory][file_name] = hennigan
        else:
            ref[directory][file_name] = "unknown"
    return ref


def apply_dan(
    ref,
    d,
):
    protocols = {
        "22_1_12 RBP-mRuby-E34 + F30-Pepx2-RBTx2-pBDTO-Com-sfGFP 535cycles_OD600_2.csv": "dan protocol 1",
        "22_2_7 mCherry and Protein E vs time with Pum2-ELP-GFP constructs.csv": "dan protocol 1, dan protocol 2",
        "22_4_12 mCherry and Pum-ELP MOPS media long term_OD600.csv": "dan protocol 3",
        "22_4_27 Com-mRuby-IDP + RNA binding targetx2 -RBS-GFP-pBDTO-Blank_OD600.csv": "dan protocol 3",
        "22_5_5 mCherry and Pum-ELP induction after confluence_OD600.csv": "dan protocol 3",
        "22_7_8 PM-ELP MPPEG and MPEG GFP and mCherry long term GFP mRuby OD600.csv": "dan protocol 3",
        "22_9_8 PM-ELP + G5P or G4rP lower induction.csv": "dan protocol 3",
        "23--23 m-peg mppegmcherrygfp2ia and 05ia.csv": "dan protocol 3",
        "23_3_14 M_PEG MstopP_PEG long bd high OD start.csv": "dan protocol 3",
        "23_3_2 M_PEG MstopP_PEG 3hr BD before induction.csv": "dan protocol 3",
        "23_4_20 M_PEG MstopP_PEG long induction low varation induction protocol.csv": "dan protocol 3",
        "23_4_22 M_PEG MstopP_PEG long induction low varation induction repeat try w yifan sterile tubes.csv": "dan protocol 3",
        "23_4_7 M_PEG MstopP_PEG long induction.csv": "dan protocol 3",
        "24_4_2_MstopP_PEG_PG_EG_G_GrowthCurvesin2xYT.csv": "dan protocol 4",
    }
    for _, _, files in os.walk(d):
        break
    
    directory = f"{d}".split(f"{str(DIR_DATA)}/experimental/")[1]
    ref[directory] = {}
    for file_name in files:
        ref[directory][file_name] = protocols[file_name]
    return ref


def apply_feilun(
    ref,
    d,
):
    for _, _, files in os.walk(d):
        break
    
    directory = f"{d}".split(f"{str(DIR_DATA)}/experimental/")[1]
    ref[directory] = {}
    for file_name in files:
        if file_name == "synth_comm_data.csv":
            ref[directory][file_name] = wu
        elif file_name == "blanksubCombinedGrowthCurvesData-DA28102T.csv":
            ref[directory][file_name] = bethke
        else:
            ref[directory][file_name] = "unknown"
    return ref


def assign_ref(
    ref,
    dir_d,
):
    for directory, source in [
        ("fujita_microbiome", fujita,),
        ("chory_lab",chory,),
        ("carolyn",zhang,),
        ("helena",ma,),
        ("huge_carin",david,),
        ("schluter",schluter,),
        ("feilunconsortia",wu,),
    ]:
        ref = apply_entire_folder_to_one(
            ref=ref,
            d=dir_d / directory,
            s=source,
        )

    ref = apply_lopatkin(
        ref=ref,
        d=dir_d / "lopatkin_lab",
    )
    ref = apply_lynch(
        ref=ref,
        d=dir_d / "lynch_lab",
    )
    ref = apply_dan(
        ref=ref,
        d=dir_d / "dan",
    )
    ref = apply_feilun(
        ref=ref,
        d=dir_d / "feilun"
    )
    for directory in [
        "zach_other",
        "dongheon",
        "rohan",
        "zach",
        "alex",
        "yuanchi",
        "ashwini",
        "cesar",
        "zhixiang",
        "sizhe",
        "katie",
        "harris",
        "grayson",
        "shangying",
        "emrah",
        "hyein",
        "jia",
        "kyeri",
    ]:
        ref = apply_entire_folder_unpublished(
            ref=ref,
            d=dir_d / directory,
        )
    return ref


zhang ="C. Zhang et al., 2020"
fujita ="Fujita et al., 2023"
baig ="Baig et al., 2023"
ma ="Ma et al., 2024"
wu ="F. Wu et al., 2022"
chory ="Chory et al., 2021"
david ="David et al., 2014"
keio ="Baba et al., 2006"
ye ="Ye et al., 2021a"
hennigan ="Hennigan et al., 2024"
li_2024 ="S. Li et al., 2024"
li_2021 ="S. Li et al., 2021"
menacho ="Menacho-Melgar et al., 2020 "
moreb ="Moreb et al., 2020"
schluter ="Schluter et al., 2020"
maddamsetti ="Maddamsetti et al., 2024"
ahmad ="Ahmad et al., 2023"
aduru ="Aduru et al., 2024"
palomino ="Palomino et al., 2023"
prensky ="Prensky et al., 2021"
bethke ="Bethke et al., 2023"


def main():

    dir_d = DIR_DATA / "experimental"
    for _, dirs, _ in os.walk(dir_d):
        break
    ref = {}
    ref = assign_ref(
        ref=ref,
        dir_d=dir_d,
    )
    for directory in dirs:
        if directory not in ref:
            print(directory)

    for_df = []

    for directory in ref.keys():
        for file_name in ref[directory].keys():
            for_df.append(
                {
                    "directory": directory,
                    "file_name": file_name,
                    "source": ref[directory][file_name],
                }
            )
    pd.DataFrame.from_records(for_df).sort_values(
        [
            "directory",
            "file_name",
        ]
    ).reset_index(drop=True).to_csv(
        DIR_RESULTS_DATA
        / "data_sources.csv",
    )
