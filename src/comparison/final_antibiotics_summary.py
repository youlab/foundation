import pandas as pd
from config import DIR_RESULTS_MODEL_COMPARISON


def print_row(
    row,
    is_header=False,
    last_row=False,
    print_first_val=False,
):
    if is_header:
        s = "\\begin{tabular}{"
        if print_first_val:
            count = len(row)
        else:
            count = len(row) - 1
        for _ in range(count-1):
            s += "c|"
        s +="c}\n\\hline\n"
    else:
        s = ""
    count = 0
    for val in row:
        count += 1
        if (count == 1) and not print_first_val:
            continue
        if pd.isnull(val):
            s += "& "
        else:
            s += f"{val} & "
    s = s[:-2]
    if not last_row:
        s += "\\\\"
    if is_header:
        s += " \hline"
    print(s)


def main():
    for prefix in [
        "classify_antibiotic_summary",
        "classify_antibiotics_CIP_summary",
        "classify_antibiotics_GM_summary",
        "classify_antibiotics_SAM_summary",
        "classify_antibiotics_SXT_summary",
        "regress_antibiotic_summary",
    ]:
        df = pd.read_csv(
            DIR_RESULTS_MODEL_COMPARISON
            / f"{prefix}_pivot.csv",
        )
        rename_columns = {}
        for i, new_col in enumerate([
            "model_type",
            "input_type",
            2,
            4,
            6,
            8,
            12,
            16,
            20,
            24,
            32,
        ]):
            rename_columns[df.columns[i]] = new_col
        df.rename(columns=rename_columns, inplace=True,)
        raw_val = str(int(df.loc[df.input_type == "raw", 2].max() * 1000))
        df = df.loc[df.input_type != "raw"].copy()
        df = df.iloc[2:].copy()
        df = df.sort_values(["model_type", "input_type",], ascending=[True, False,]).reset_index(drop=True)
        df.drop(columns="input_type", inplace=True)
        print(f"\n\nBeginning table for {prefix}, raw curves accuracy = {float(raw_val) / 1000}")
        print_row(
            row=df.columns,
            is_header=True,
            print_first_val=True,
        )
        for idx, row in df.iterrows():
            print_row(
                row=row,
                last_row=idx==(df.shape[0] - 1),
                print_first_val=True,
            )
        print("\end{tabular}")
        df.to_csv(
            DIR_RESULTS_MODEL_COMPARISON
            / f"{prefix}_final_{raw_val}.csv",
        )