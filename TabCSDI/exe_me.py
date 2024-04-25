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

from src.main_model_table import TabCSDI
from src.utils_table import train, evaluate
from dataset_me_onehot import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--config", type=str, default="data.yaml")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.8)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
print(args)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

# Create folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/breast_fold" + str(args.nfold) + "_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = TabCSDI(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + "breast_fold5_20240416_052953" + "/model.pth"))
print("---------------Start testing---------------")
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
#all_target = []
#all_generated_samples = []
#with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
#    for batch_no, test_batch in enumerate(it, start=1):
#        output = model.evaluate(test_batch, 100)
#        samples, c_target, eval_points, observed_points, observed_time = output
#        samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
#        c_target = c_target.permute(0, 2, 1)
#        all_target.append(c_target)
#        all_generated_samples.append(samples)
#all_generated_samples = torch.cat(all_generated_samples, dim=0)
#all_target = torch.cat(all_target, dim=0)

#with open(
#    foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
#    ) as f:
#        pickle.dump([all_generated_samples,all_target], f,)

