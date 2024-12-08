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
from match_state_ft import match_state
from main_model_table import TabCSDI
from utils_table import train, evaluate
from dataset_me_ft import get_dataloader

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

path = "./" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

foldernames = ["/SRS", "/RS", "/WRS", "/SFS", "/ODS_extTail", "/SSRS_exactAlloc", 
               "/ODS_exactAlloc", "/RS_exactAlloc", "/WRS_exactAlloc", "/SFS_exactAlloc"]
#foldernames = ["/SFS_exactAlloc"]

for i in range(args.startdigit, args.enddigit + 1):
    k = str(i).zfill(4)
    for j in foldernames:
        foldername = "./save/model" + "_" + k + j + "/"
        print("model folder:", foldername)
        os.makedirs(foldername, exist_ok=True)
        with open(foldername + "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        
        path = "/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample" + j + j + "_" + k + ".csv"
        if j in ["/SRS", "/RS", "/WRS", "/SFS", "/ODS_extTail"]:
            batch_loader, dataset, tokenizer, tokenized_values = get_dataloader(path = path,
                                                   batch_size = config["train"]["batch_size"],
                                                   token_dim = config["model"]["token_emb_dim"],
                                                   categorical_cols = ["idx"],
                                                   binary_cols = ["R", "usborn", "high_chol", 
                                                                  "female", "bkg_pr", "bkg_o", "hypertension"],
                                                   phase1_cols = ["c_ln_na_bio1", "c_ln_k_bio1", 
                                                                  "c_ln_kcal_bio1", "c_ln_protein_bio1"],
                                                   phase2_cols = ["c_ln_na_true", "c_ln_k_true", 
                                                                  "c_ln_kcal_true", "c_ln_protein_true"],
                                                   strata_col = "idx")
        elif j == "/SSRS_exactAlloc":
            batch_loader, dataset, tokenizer, tokenized_values = get_dataloader(path = path,
                                                   batch_size = config["train"]["batch_size"],
                                                   token_dim = config["model"]["token_emb_dim"],
                                                   categorical_cols = ["idx"],
                                                   binary_cols = ["R", "usborn", "high_chol", 
                                                                  "female", "bkg_pr", "bkg_o", "hypertension"],
                                                   phase1_cols = ["c_ln_na_bio1", "c_ln_k_bio1", 
                                                                  "c_ln_kcal_bio1", "c_ln_protein_bio1"],
                                                   phase2_cols = ["c_ln_na_true", "c_ln_k_true", 
                                                                  "c_ln_kcal_true", "c_ln_protein_true"],
                                                   strata_col = "idx")
        elif j == "/ODS_exactAlloc":
            batch_loader, dataset, tokenizer, tokenized_values = get_dataloader(path = path,
                                                   batch_size = config["train"]["batch_size"],
                                                   token_dim = config["model"]["token_emb_dim"],
                                                   categorical_cols = ["idx", "outcome_strata"],
                                                   binary_cols = ["R", "usborn", "high_chol", 
                                                                  "female", "bkg_pr", "bkg_o", "hypertension"],
                                                   phase1_cols = ["c_ln_na_bio1", "c_ln_k_bio1", 
                                                                  "c_ln_kcal_bio1", "c_ln_protein_bio1"],
                                                   phase2_cols = ["c_ln_na_true", "c_ln_k_true", 
                                                                  "c_ln_kcal_true", "c_ln_protein_true"],
                                                   strata_col = "outcome_strata")
        elif j == "/SFS_exactAlloc":
            batch_loader, dataset, tokenizer, tokenized_values = get_dataloader(path = path,
                                                   batch_size = config["train"]["batch_size"],
                                                   token_dim = config["model"]["token_emb_dim"],
                                                   categorical_cols = ["idx", "sfs_strata"],
                                                   binary_cols = ["R", "usborn", "high_chol", 
                                                                  "female", "bkg_pr", "bkg_o", "hypertension"],
                                                   phase1_cols = ["c_ln_na_bio1", "c_ln_k_bio1", 
                                                                  "c_ln_kcal_bio1", "c_ln_protein_bio1"],
                                                   phase2_cols = ["c_ln_na_true", "c_ln_k_true", 
                                                                  "c_ln_kcal_true", "c_ln_protein_true"],
                                                   strata_col = "sfs_strata")
            
        elif j == "/RS_exactAlloc":
            batch_loader, dataset, tokenizer, tokenized_values = get_dataloader(path = path,
                                                   batch_size = config["train"]["batch_size"],
                                                   token_dim = config["model"]["token_emb_dim"],
                                                   categorical_cols = ["idx", "rs_strata"],
                                                   binary_cols = ["R", "usborn", "high_chol", 
                                                                  "female", "bkg_pr", "bkg_o", "hypertension"],
                                                   phase1_cols = ["c_ln_na_bio1", "c_ln_k_bio1", 
                                                                  "c_ln_kcal_bio1", "c_ln_protein_bio1"],
                                                   phase2_cols = ["c_ln_na_true", "c_ln_k_true", 
                                                                  "c_ln_kcal_true", "c_ln_protein_true"],
                                                   strata_col = "rs_strata")
        elif j == "/WRS_exactAlloc":
            batch_loader, dataset, tokenizer, tokenized_values = get_dataloader(path = path,
                                                   batch_size = config["train"]["batch_size"],
                                                   token_dim = config["model"]["token_emb_dim"],
                                                   categorical_cols = ["idx", "wrs_strata"],
                                                   binary_cols = ["R", "usborn", "high_chol", 
                                                                  "female", "bkg_pr", "bkg_o", "hypertension"],
                                                   phase1_cols = ["c_ln_na_bio1", "c_ln_k_bio1", 
                                                                  "c_ln_kcal_bio1", "c_ln_protein_bio1"],
                                                   phase2_cols = ["c_ln_na_true", "c_ln_k_true", 
                                                                  "c_ln_kcal_true", "c_ln_protein_true"],
                                                   strata_col = "wrs_strata")

        processed_data_path = f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}_zscore_norm.pk"

        min_t, p1, p2 = match_state(dataset, tokenized_values,
                            config=config, device=args.device)
        model = TabCSDI(config, args.device, 
                        target_dim = (dataset.observed_values).shape[1],
                        matched_state = [x + 1 for x in min_t], 
                        tokenizer = tokenizer,
                        phase1_cols = p1,
                        phase2_cols = p2,
                        cont_list = dataset.cont_cols,
                        num_cate_list = dataset.num_cate_cols).to(args.device)

        train(model,
              config["train"],
              batch_loader,
              foldername=foldername
        )

        print("---------------Start testing---------------")
        evaluate(model, tokenizer, batch_loader, 
                nsample = args.nsample,
                foldername = foldername, 
                filename = j + "_" + k, 
                subfolder = j)
