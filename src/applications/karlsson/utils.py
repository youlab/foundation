from pathlib import Path
from datetime import datetime
import re
import os
import csv
import json

import numpy as np
import pandas as pd


def _read_text(path: str):
    '''read text with correct encoding'''
    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            return Path(path).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(errors="replace")


def _canonical_treatment_name(filename: str):
    '''return canonical treatment name from dataset file name'''
    s = filename.lower()
    if "dark" in s:
        return "dark"
    if "420" in s:
        return "420"
    if "450" in s:
        return "450"
    if "530" in s:
        return "530"
    if "620" in s:
        return "620"
    if "660" in s:
        return "660"
    return "unknown"


def _canonical_plate_type(plate_type: str):
    '''return canonical plate type name'''
    s = (plate_type or "").replace(" ", "").upper()
    if s.startswith("PM1"):
        return "PM1"
    if s.startswith("PM2"):
        return "PM2"
    return "unknown"


def _canonical_setup_time(setup_time: str):
    '''return canonical setup time'''
    if setup_time is None or str(setup_time).strip() == "":
        return pd.NaT

    try:
        dt = datetime.strptime(setup_time.strip(), "%b %d %Y %I:%M %p")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return setup_time


def _canonical_sample_number(sample_number: str):
    '''return dict with canonical sample labeling information'''

    raw = "" if sample_number is None else str(sample_number)
    clean = re.sub(r"\s+", "", raw)
    low = clean.lower()

    if low.startswith("pchl"):
        prefix = "Pchl"
        organism = "Pseudomonas chlororaphis"
        rest = clean[4:]
    elif low.startswith("pc"):
        prefix = "Pc"
        organism = "Pseudomonas chlororaphis"
        rest = clean[2:]
    elif low.startswith("ba"):
        prefix = "Ba"
        organism = "Bacillus amyloliquefaciens"
        rest = clean[2:]
    elif low.startswith("sg"):
        prefix = "Sg"
        organism = "Streptomyces griseoviridis"
        rest = clean[2:]
    else:
        prefix = "unknown"
        organism = "unknown"
        rest = clean

    m = re.search(r"([1-6])", rest)
    replicate = int(m.group(1)) if m else np.nan

    return {
        "sample_number_raw": raw,
        "sample_number_clean": clean,
        "sample_prefix": prefix,
        "organism": organism,
        "replicate": replicate,
    }


def parse_omnilog_csv(path: str):
    '''parse entire OmniLog output .csv file'''
    treatment = _canonical_treatment_name(Path(path).name)
    text = _read_text(path)
    lines = text.splitlines()

    trajectories = []
    i = 0
    block_i = 0

    while i < len(lines):
        if not lines[i].startswith("Data File"):
            i += 1
            continue

        # parse metadata until the Hour header
        meta = {}
        j = i
        while j < len(lines) and not lines[j].lstrip().startswith("Hour"):
            parts = next(csv.reader([lines[j]]))
            if parts:
                key = parts[0].strip()
                val = parts[1].strip() if len(parts) > 1 else ""
                if key:
                    meta[key] = val
            j += 1

        if j >= len(lines):
            break

        # parse header row
        header = [h.strip() for h in next(csv.reader([lines[j]]))]
        if not header or header[0].lower() != "hour":
            i = j + 1
            continue

        # parse data rows until empty or next block
        data_rows = []
        k = j + 1
        while k < len(lines):
            line = lines[k]
            if not line.strip() or set(line.strip()) == {","} or line.startswith("Data File"):
                break
            row = [x.strip() for x in next(csv.reader([line]))]
            if not row or not row[0]:
                break
            data_rows.append(row)
            k += 1

        well_cols = [h for h in header[1:] if h]
        
        # parse metadata
        sample_info = _canonical_sample_number(meta.get("Sample Number", ""))
        setup_ts = _canonical_setup_time(meta.get("Set up Time", ""))
        plate_type = _canonical_plate_type(meta.get("Plate Type", ""))

        # convert block table to csv like format
        if data_rows:
            block_df = pd.DataFrame(data_rows, columns=header[:len(data_rows[0])])
            block_df = block_df.rename(columns={block_df.columns[0]: "within_window_hour"})
            block_df["within_window_hour"] = pd.to_numeric(block_df["within_window_hour"], errors="coerce")
            
            for well in well_cols:
                block_df[well] = pd.to_numeric(block_df[well], errors="coerce")

            for well in well_cols:
                times = block_df["within_window_hour"].values
                values = block_df[well].values
                
                # only include trajectories with valid data
                valid_mask = ~np.isnan(values)
                if valid_mask.sum() > 1:  # at least 2 valid points
                    times_valid = times[valid_mask]
                    values_valid = values[valid_mask]

                    # normalize
                    max_value = np.max(values_valid)
                    values_normalized = values_valid / max_value
                    
                    rep_label = int(sample_info["replicate"]) if pd.notna(sample_info["replicate"]) else "NA"
                    trajectory_id = f"{treatment}_{sample_info['organism']}_rep{rep_label}_{plate_type}_{well}"
                    
                    trajectories.append({
                        "trajectory_id": trajectory_id,
                        "treatment": treatment,
                        "organism": sample_info["organism"],
                        "replicate": sample_info["replicate"],
                        "plate_type": plate_type,
                        "well": well,
                        "setup_time": setup_ts,
                        "source_file": Path(path).name,
                        "block_id": f"{treatment}_{Path(path).stem}_{block_i:04d}",
                        "time_points": times_valid.tolist(), # in hours
                        "trajectory": values_valid.tolist(),
                        "trajectory_normalized": values_normalized.tolist(),
                        "max_value": float(max_value),
                        "n_points": int(len(values_valid)),
                    })

        block_i += 1
        i = max(k, j + 1)

    return trajectories