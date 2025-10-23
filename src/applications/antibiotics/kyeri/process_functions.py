antibiotics = {
    "cefo": "cefotaxime",
    "amx": "amoxicillin",
    "carb": "carbenicillin",
}


def process1(df):
    df["cas"] = [float(x[0]) if x[1].lower() == "cas" else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split(" ")]
    df["glu"] = [float(x[0]) if x[1].lower() == "glu" else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split(" ")]
    df["cm"] = [float(x[0]) if x[1].lower() == "cm" else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split(" ")]
    df["antibiotic_conc"] = [float(x[3]) if x[2].lower() == "carb" else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split(" ")]
    df["antibiotic_type"] = "none"
    df.loc[(df.loc[:, "antibiotic_conc"] > 0) & (df.loc[:, "cm"] == 0), "antibiotic_type"] = "carbenicillin"
    df.loc[(df.loc[:, "antibiotic_conc"] == 0) & (df.loc[:, "cm"] > 0), "antibiotic_type"] = "chloramphenicol"
    df.loc[(df.loc[:, "antibiotic_conc"] > 0) & (df.loc[:, "cm"] > 0), "antibiotic_type"] = "carbenicillin/chloramphenicol"
    return df


def process2(df):
    df["cas"] = [float(x[0].split("cas")[1]) if x[0].find("cas") != -1 else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("+")]
    df["antibiotic_conc"] = [float(x[-1].split("carb")[1]) if x[-1].find("carb") != -1 else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("+")]
    df["cm"] = [float(x[1].split("cm")[1]) if x[1].find("cm") != -1 else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("+")]
    
    df["antibiotic_type"] = "none"
    df.loc[(df.loc[:, "antibiotic_conc"] > 0) & (df.loc[:, "cm"] == 0), "antibiotic_type"] = "carbenicillin"
    df.loc[(df.loc[:, "antibiotic_conc"] == 0) & (df.loc[:, "cm"] > 0), "antibiotic_type"] = "chloramphenicol"
    df.loc[(df.loc[:, "antibiotic_conc"] > 0) & (df.loc[:, "cm"] > 0), "antibiotic_type"] = "carbenicillin/chloramphenicol"
    return df


def process3(df):
    df["cas"] = [float(x[0].split("Cas ")[1]) if x[0].lower().find("cas") != -1 else 0. for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("_")]
    df["antibiotic_conc"] = [float(x[-1].split(" ")[1]) for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("_")]
    df["antibiotic_type"] = [antibiotics[x[-1].split(" ")[0].lower()] for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("_")]
    df.loc[df["antibiotic_conc"] == 0, "antibiotic_type"] = "none"
    return df


def process4(df):
    df["plasmid"] = [x[0] for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("-")]
    df["antibiotic_type"] = "carbenicillin"
    df["antibiotic_conc"] = [float(x[2].split("Carb")[1]) for x in df.loc[:, "casaa%_antibiotic_conc"].str.split("-")]
    
    df.loc[df["antibiotic_conc"] == 0, "antibiotic_type"] = "none"
    return df


def process5(df):
    df = df[df.Strain != "media_blank"].copy()
    df["antibiotic_conc"] = [float(x[0]) for x in df.antibiotic_conc.str.split("ug/ml")]
    df.loc[df["antibiotic_conc"] == 0, "antibiotic_type"] = "none"
    df["Reading"] = "OD600"
    return df


def process6(df):
    df["antibiotic_conc"] = [float(x[-2].split("Carb")[1]) if x[-2].lower().find("carb") != -1 else float(x[-2].split("amx")[1]) for x in df.loc[:, "conditions_dilutionfactor"].str.split("_")]
    df["antibiotic_type"] = ["carbenicillin" if x[-2].lower().find("carb") != -1 else antibiotics["amx"] for x in df.loc[:, "conditions_dilutionfactor"].str.split("_")]
    
    df.loc[df["antibiotic_conc"] == 0, "antibiotic_type"] = "none"
    df["Reading"] = "OD600"
    return df
