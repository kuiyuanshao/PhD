import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import pandas as pd
from datetime import datetime
import os


def train(
    model,
    config,
    train_loader,
    foldername="",
):
    # Control random seed in the current script.
    torch.manual_seed(0)
    np.random.seed(0)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p0 = int(0.25 * config["epochs"])
    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    p3 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    # history = {'train_loss':[], 'val_rmse':[]}
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                # The forward method returns loss.
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()


    if foldername != "":
        torch.save(model.state_dict(), output_path)


def evaluate(model, tokenizer, test_loader, nsample=100,
             foldername="", filename="", subfolder="", file_to_del=""):
    with open("./processed_data" + filename + "/transformed_columns.pk", "rb") as f:
        cont_list, num_cate_list = pickle.load(f)
    with open("./processed_data" + filename + "/encoder.pk", "rb") as f:
        encoder = pickle.load(f)
    with torch.no_grad():
        model.eval()
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points = output
                samples = tokenizer.recover(samples, len(cont_list))
                B, L = samples.shape
                all_generated_samples.append(samples.reshape(B, 1, L))
                
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            imputations = all_generated_samples.cpu().numpy()
            foldername2 = "./imputations/" + subfolder
            os.makedirs(foldername2, exist_ok=True)
                
            with pd.ExcelWriter(foldername2 + filename + ".xlsx") as writer:
                for i in range(imputations.shape[1]):
                    df = pd.DataFrame(imputations[:, i, :])
                    df = df.to_numpy()
                    df = encoder.inverse_transform(df)
                    pd.DataFrame(df).to_excel(writer, sheet_name=f"Sheet_{i+1}", index=False)
                        
            #for file in file_to_del:
            #    os.remove(file)
            #    print(f"Deleted: {file}")

