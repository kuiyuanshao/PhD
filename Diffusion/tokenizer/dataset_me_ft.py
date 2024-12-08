# This script is for generating .pk file for mixed data types dataset
import pickle
import yaml
import os
import math
import re
import numpy as np
import pandas as pd
import category_encoders as ce
import torch

from torch.utils.data import DataLoader, Dataset
from tokenizer import Tokenizer


def process_func(path: str, categorical_cols: str, phase1_cols: str, phase2_cols: str, strata_col: str):
    data = pd.read_csv(path).iloc[:, 1:]
    data = pd.concat([data] * 1)
    
    categorical_cols = [data.columns.get_loc(col) for col in categorical_cols]
    # Swap columns
    temp_list = [i for i in range(data.shape[1]) if i not in categorical_cols] #Variables not in the categorical list
    temp_list.extend(categorical_cols) # append the categorical col indices to the end of the temp list
    new_cols_order = temp_list
    data = data.reindex(columns=data.columns[new_cols_order])
    strata_info = data.get(strata_col)
    phase1_cols = [data.columns.get_loc(col) for col in phase1_cols]
    phase2_cols = [data.columns.get_loc(col) for col in phase2_cols]   
    data.columns = [i for i in range(data.shape[1])]

    # create two lists to store position
    cont_list = [i for i in range(0, data.shape[1] - len(categorical_cols))]
    cat_list = [i for i in range(len(cont_list), data.shape[1])]

    observed_values = data.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)
    
    masks = observed_masks.copy()
    gt_masks = masks.reshape(observed_masks.shape)

    num_cate_list = []
    # set encoder here
    encoder = ce.ordinal.OrdinalEncoder(cols=data.columns[cat_list])
    encoder.fit(data)
    new_df = encoder.transform(data)
    # we now need to transform these masks to the new one, suitable for mixed data types.
    cum_num_bits = 0
    new_observed_masks = observed_masks.copy()
    new_gt_masks = gt_masks.copy()

    for index, col in enumerate(cat_list):
        num_cate_list.append(new_df.iloc[:, col].nunique())
        corresponding_cols = len(
            [
                s
                for s in new_df.columns
                if isinstance(s, str) and s.startswith(str(col) + "_")
            ]
        )
        add_col_num = corresponding_cols
        insert_col_obs = observed_masks[:, col]
        insert_col_gt = gt_masks[:, col]

        for i in range(add_col_num - 1):
            new_observed_masks = np.insert(
                new_observed_masks, cum_num_bits + col, insert_col_obs, axis=1
            )
            new_gt_masks = np.insert(
                new_gt_masks, cum_num_bits + col, insert_col_gt, axis=1
            )
        cum_num_bits += add_col_num - 1
    new_observed_values = new_df.values
    new_observed_values = np.nan_to_num(new_observed_values)
    new_observed_values = new_observed_values.astype(np.float64)

    os.makedirs("./processed_data/" + str(os.path.splitext(os.path.basename(path))[0]), exist_ok=True)

    with open("./processed_data/" + str(os.path.splitext(os.path.basename(path))[0]) + "/transformed_columns.pk", "wb") as f:
        pickle.dump([cont_list, num_cate_list], f)

    with open("./processed_data/" + str(os.path.splitext(os.path.basename(path))[0]) +"/encoder.pk", "wb") as f:
        pickle.dump(encoder, f)

   
    return new_observed_values, new_observed_masks, new_gt_masks, cont_list, num_cate_list, phase1_cols, phase2_cols, strata_info
    

class tabular_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(self, path="", categorical_cols="", phase1_cols="", phase2_cols="", strata_col=""):
        processed_data_path = f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}.pk"
        processed_data_path_norm = f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}_zscore_norm.pk"
        os.makedirs("processed_data", exist_ok=True)
        
        self.observed_values, self.observed_masks, self.gt_masks, self.cont_cols, self.num_cate_cols, self.phase1_cols, self.phase2_cols, self.strata_info = process_func(path, categorical_cols=categorical_cols, phase1_cols=phase1_cols, phase2_cols=phase2_cols, strata_col=strata_col)
        with open(processed_data_path, "wb") as f:
            pickle.dump([self.observed_values, self.observed_masks, self.gt_masks,
                         self.cont_cols, self.num_cate_cols, self.phase1_cols, self.phase2_cols, self.strata_info], f)
        print("--------Dataset created--------")
        
        col_num = len(self.cont_cols)
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        std_arr = np.zeros(col_num)
        
        for k in range(col_num):
            obs_ind = self.observed_masks[:, k].astype(bool)
            temp = self.observed_values[:, k]
            max_arr[k] = max(temp[obs_ind])
            min_arr[k] = min(temp[obs_ind])
            mean_arr[k] = np.mean(temp[obs_ind])
            std_arr[k] = np.std(temp[obs_ind])
            
        cont_phase1_cols = [col for col in self.phase1_cols if col in self.cont_cols]
        cont_phase2_cols = [col for col in self.phase2_cols if col in self.cont_cols]
        max_arr[cont_phase1_cols] = max_arr[cont_phase2_cols]
        min_arr[cont_phase1_cols] = min_arr[cont_phase2_cols]
        std_arr[cont_phase1_cols] = std_arr[cont_phase2_cols]
        mean_arr[cont_phase1_cols] = mean_arr[cont_phase2_cols]
        
        print(f"--------------Phase-1 Columns MEANS{mean_arr[cont_phase2_cols]}--------------")
        print(f"--------------Phase-2 Columns STDS{std_arr[cont_phase2_cols]}--------------")
        print(f"--------------Phase-1 Columns {self.phase1_cols}--------------")
        print(f"--------------Phase-2 Columns {self.phase2_cols}--------------")
        
        for index, k in enumerate(self.cont_cols):
            self.observed_values[:, k] = (self.observed_values[:, k] - mean_arr[k] + 1e-6) / (std_arr[k] + 1e-6)

        with open(processed_data_path_norm, "wb") as f:
            pickle.dump([self.observed_values, self.observed_masks, self.gt_masks, 
                         self.cont_cols, self.num_cate_cols, self.phase1_cols, self.phase2_cols, self.strata_info], f)

        self.use_index_list = np.arange(len(self.observed_values))

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(path="", batch_size=128, token_dim = 8, categorical_cols="", binary_cols="", phase1_cols="", phase2_cols="", strata_col="", device = "cuda"):
    categorical_cols = categorical_cols + binary_cols
    dataset = tabular_dataset(path=path, 
                              categorical_cols=categorical_cols,
                              phase1_cols=phase1_cols, 
                              phase2_cols=phase2_cols, 
                              strata_col=strata_col)
    batch_loader = DataLoader(dataset, batch_size = batch_size, shuffle = 0)
    tokenizer = Tokenizer(len(dataset.cont_cols), dataset.num_cate_cols, token_dim, False,
                          np.intersect1d(dataset.phase1_cols, dataset.cont_cols),
                          np.intersect1d(dataset.phase2_cols, dataset.cont_cols),
                          np.setdiff1d(dataset.phase1_cols, dataset.cont_cols)-len(dataset.cont_cols),
                          np.setdiff1d(dataset.phase2_cols, dataset.cont_cols)-len(dataset.cont_cols)).to(device)
    observed_values = torch.from_numpy(dataset.observed_values).to(device)
    B, K = observed_values.shape
    observed_values = observed_values.reshape(B, 1, K)
    tokenized_values = tokenizer(observed_values[:, :, dataset.cont_cols], observed_values[:, :, len(dataset.cont_cols):])
    B, K, L, C = tokenized_values.shape
    #Embeddings are flattened.
    tokenized_values = tokenized_values.reshape(B, L * C)
    return batch_loader, dataset, tokenizer, tokenized_values