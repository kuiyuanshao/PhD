import argparse
import torch
import datetime
import json
import yaml
import os

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import logging
import scipy.stats as stats
from matplotlib import pyplot as plt
from functools import partial
from scipy.stats import norm
import math
from numpy import genfromtxt
from match_state import match_state
from src.main_model_table import TabCSDI
from src.utils_table import train, evaluate
from dataset_me_onehot import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--config", type=str, default="nutritional.yaml")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--startdigit", type=int, default=1)
parser.add_argument("--enddigit", type=int, default=101)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

foldernames = ["/-1_0_-2_0_-0.25", "/-1_0_-2_0_0", "/-1_0_-2_0_0.25",
               "/-1_0.25_-2_0.5_-0.25", "/-1_0.25_-2_0.5_0", "/-1_0.25_-2_0.5_0.25",
               "/-1_0.5_-2_1_-0.25", "/-1_0.5_-2_1_0", "/-1_0.5_-2_1_0.25",
               "/-1_1_-2_2_-0.25", "/-1_1_-2_2_0", "/-1_1_-2_2_0.25"]
designnames = ["/SRS", "/BLS"]
#foldernames = ["/SFS_exactAlloc"]

for i in range(args.startdigit, args.enddigit + 1):
    k = str(i).zfill(4)
    for j in foldernames:
        for m in designnames:
            foldername = "./save/model_" + k + j + m + "/"
            print("model folder:", foldername)
            os.makedirs(foldername, exist_ok=True)
            with open(foldername + "config.json", "w") as f:
                json.dump(config, f, indent=4)


            path = "/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/SurvivalSample" + j + m + m + "_" + k + ".csv"
            if m == "/SRS":
                batch_loader, dataset = get_dataloader(path = path,
                                                       batch_size = config["train"]["batch_size"],
                                                       categorical_cols = ["CFAR_PID"],
                                                       binary_cols = ["A.star", "D.star", "C.star", "ade.star", "A", "D", "C", "ade"],
                                                       phase1_cols = ["A.star", "D.star", "lastC.star", 
                                                          "FirstOImonth.star", "FirstARTmonth.star",
                                                          "AGE_AT_LAST_VISIT.star", "C.star", 
                                                          "ARTage.star", "OIage.star", "last.age.star", 
                                                          "ade.star", "fu.star"],
                                                       phase2_cols = ["A", "D", "lastC", 
                                                          "FirstOImonth", "FirstARTmonth",
                                                          "AGE_AT_LAST_VISIT", "C", 
                                                          "ARTage", "OIage", "last.age", 
                                                          "ade", "fu"],
                                                       strata_col = "")
            elif m == "/BLS":
                batch_loader, dataset = get_dataloader(path = path,
                                                       batch_size = config["train"]["batch_size"],
                                                       categorical_cols = ["CFAR_PID"],
                                                       binary_cols = ["A.star", "D.star", "C.star", "ade.star", "A", "D", "C", "ade"],
                                                       phase1_cols = ["A.star", "D.star", "lastC.star", 
                                                          "FirstOImonth.star", "FirstARTmonth.star",
                                                          "AGE_AT_LAST_VISIT.star", "C.star", 
                                                          "ARTage.star", "OIage.star", "last.age.star", 
                                                          "ade.star", "fu.star"],
                                                       phase2_cols = ["A", "D", "lastC", 
                                                          "FirstOImonth", "FirstARTmonth",
                                                          "AGE_AT_LAST_VISIT", "C", 
                                                          "ARTage", "OIage", "last.age", 
                                                          "ade", "fu"],
                                                       strata_col = "Strata")
            

            processed_data_path = f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}_max-min_norm.pk"

            min_t = match_state(processed_data_path, config)
            model = TabCSDI(config, args.device, target_dim = (dataset.observed_values).shape[1],
                            matched_state = [x + 1 for x in min_t], 
                            phase1_cols = dataset.phase1_cols,
                            phase2_cols = dataset.phase2_cols).to(args.device)
            train(model,
                config["train"],
                batch_loader,
                foldername=foldername
            )

            print("---------------Start testing---------------")
            file_to_del = [f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}.pk", 
                           f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}_max-min_norm.pk"]
            evaluate(model, batch_loader, 
                    nsample = args.nsample,
                    foldername = foldername, 
                    filename = m + "_" + k, 
                    subfolder = j + m,
                    file_to_del = file_to_del)
