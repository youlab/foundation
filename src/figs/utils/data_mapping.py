import numpy as np

from data.utils import get_data


def get_mapping_and_samples_for_experimental(
    consortia_names,
):
    category = "experimental"
    data, idx = get_data(category=category)
    samples = {}
    
    keys = sorted(list(idx[category].keys()))

    names = []
    for key in keys:
        name = key.split("_")[0]
        if name not in names:
            names.append(name)
    
    for name in names:
        samples[name] = []
    
    for key in keys:
        i0 = idx[category][key]["y_all_i"]
        i1 = i0 + idx[category][key]["y.shape"][0]
        if i0 == i1:
            continue
    
        samples[key.split("_")[0]].extend(np.arange(i0, i1).tolist())

    mapping = {}
    i_clonal = 0
    i_consortia = 24
    for key in samples.keys():
        if key in consortia_names:
            mapping[i_consortia] = key
            i_consortia += 1
        else:
            mapping[i_clonal] = key
            i_clonal += 1
    
    return data, samples, mapping


def get_mapping_and_samples_for_simulation(
    consortia_names,
):
    category = "simulation"
    data, idx = get_data(category=category)
    samples = {}
    
    keys = sorted(list(idx[category].keys()))

    names = []
    for key in keys:
        name = key.split("_")[0]
        if name not in names:
            names.append(name)
    
    for name in names:
        samples[name] = []
    
    for key in keys:
        i0 = idx[category][key]["y_all_i"]
        i1 = i0 + idx[category][key]["y.shape"][0]
        if i0 == i1:
            continue
    
        samples[key.split("_")[0]].extend(np.arange(i0, i1).tolist())

    mapping = {}
    i_clonal = 0
    i_consortia = 2
    for key in samples.keys():
        if key in consortia_names:
            mapping[i_consortia] = key
            i_consortia += 1
        else:
            mapping[i_clonal] = key
            i_clonal += 1
    
    return data, samples, mapping
