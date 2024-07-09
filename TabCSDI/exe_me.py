import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.main_model_table import TabCSDI
from src.utils_table import train, evaluate
from dataset_me_onehot import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--config", type=str, default="data.yaml")
parser.add_argument("--device", default="cpu", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

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
foldername = "./save/datald" + str(args.nfold) + "_" + current_time + "/"
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

#if args.modelfolder == "":
#    train(
#        model,
#        config["train"],
#        train_loader,
#        valid_loader=valid_loader,
#        foldername=foldername,
#    )
#else:
model.load_state_dict(torch.load("./save/" + "datald5_20240410_133615" + "/model.pth"))
print("---------------Start testing---------------")
#evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
output = []
with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
    for batch_no, test_batch in enumerate(it, start=1):
        curr = model.evaluate(test_batch, 10)[0]
        curr = curr.permute(0, 1, 3, 2)
        samples_median = curr.median(dim=1)
        output.append(samples_median.values.numpy())
        
output = np.concatenate(np.concatenate(np.concatenate(output, axis=0), axis=0), axis=0)
pd.DataFrame(output).to_csv("generated_data.csv", sep = "\t")
