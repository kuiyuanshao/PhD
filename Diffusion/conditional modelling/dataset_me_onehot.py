import pickle
import yaml
import os
import os.path
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

def process_func(path: str, categorical_cols: str, binary_cols: str, phase1_cols: str, phase2_cols: str, strata_col: str):
    data = pd.read_csv(path).iloc[:, 1:]
    data_aug = pd.concat([data] * 1)
    
    strata_info = data_aug.get(strata_col)
    
    if categorical_cols != "":
        #for col in categorical_cols:
        #    data_aug[col] = pd.factorize(data_aug[col])[0]
        data_aug = pd.get_dummies(data_aug, columns = categorical_cols)
    binary_cols = [data_aug.columns.get_loc(col) for col in binary_cols]
    phase1_cols = [data_aug.columns.get_loc(col) for col in phase1_cols]
    phase2_cols = [data_aug.columns.get_loc(col) for col in phase2_cols]   
    
    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)
    
    masks = observed_masks.copy()
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype(int)
    gt_masks = gt_masks.astype(int)
    
    return observed_values, observed_masks, gt_masks, phase1_cols, phase2_cols, strata_info, binary_cols

class tabular_dataset(Dataset):
    def __init__(self, path="", categorical_cols="", binary_cols="", phase1_cols="", phase2_cols="", strata_col=""):
        processed_data_path = f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}.pk"
        processed_data_path_norm = f"./processed_data/{os.path.splitext(os.path.basename(path))[0]}_zscore_norm.pk"
        os.makedirs("processed_data", exist_ok=True)

        self.observed_values, self.observed_masks, self.gt_masks, self.phase1_cols, self.phase2_cols, self.strata_info, self.binary_cols = process_func(path, categorical_cols, binary_cols, phase1_cols, phase2_cols, strata_col)

        with open(processed_data_path, "wb") as f:
            pickle.dump([self.observed_values, self.observed_masks, self.gt_masks, 
                         self.phase1_cols, self.phase2_cols, self.strata_info], f)
            print("--------Dataset created--------")
        col_num = self.observed_values.shape[1]
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        std_arr = np.zeros(col_num)
        #self.observed_values = resample_phase2(self.observed_values, self.strata_info, self.phase2_cols)
        for k in range(col_num):
            obs_ind = self.observed_masks[:, k].astype(bool)
            temp = self.observed_values[:, k]
            max_arr[k] = max(temp[obs_ind])
            min_arr[k] = min(temp[obs_ind])
            mean_arr[k] = np.mean(temp[obs_ind])
            std_arr[k] = np.std(temp[obs_ind])
            
        
        max_arr[self.phase1_cols] = max_arr[self.phase2_cols]
        min_arr[self.phase1_cols] = min_arr[self.phase2_cols]
        std_arr[self.phase1_cols] = std_arr[self.phase2_cols]
        mean_arr[self.phase1_cols] = mean_arr[self.phase2_cols]

        print(f"--------------Phase-1 Columns {self.phase1_cols}--------------")
        print(f"--------------Phase-2 Columns {self.phase2_cols}--------------")
        for k in range(col_num):
            if k in self.binary_cols:
                self.observed_values[:, k] = (2 * (self.observed_values[:, k]) - 1) * self.observed_masks[:, k]
            else:
                self.observed_values[:, k] = (self.observed_values[:, k] + 1e-8 - mean_arr[k]) / (std_arr[k] + 1e-8) * self.observed_masks[:, k]
                #self.observed_values[:, k] = ((self.observed_values[:, k] - (min_arr[k] - 1)) / (max_arr[k] - min_arr[k] + 1)) * self.observed_masks[:, k]
                #self.observed_values[:, k] = ((self.observed_values[:, k] - 0 + 1) / (max_arr[k] - 0 + 1)) * self.observed_masks[:, k]


        with open(processed_data_path_norm, "wb") as f:
            pickle.dump([self.observed_values, self.observed_masks, self.gt_masks, 
                         self.phase1_cols, self.phase2_cols, self.strata_info, self.binary_cols], f)
        if os.path.isfile(processed_data_path_norm):
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, self.phase1_cols, self.phase2_cols, self.strata_info, self.binary_cols = pickle.load(f)
            print("--------Normalized dataset loaded--------")
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

def get_dataloader(path="", batch_size=128, categorical_cols="", binary_cols="", phase1_cols="", phase2_cols="", strata_col=""):
    dataset = tabular_dataset(path=path, 
                              categorical_cols=categorical_cols, 
                              binary_cols=binary_cols,
                              phase1_cols=phase1_cols, 
                              phase2_cols=phase2_cols, 
                              strata_col=strata_col)
    batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=0)
    return batch_loader, dataset
    

def resample_phase2(df, strata_info, phase2_cols):
    if strata_info is not None:
        unique_strata = np.unique(strata_info)
        for stratum in unique_strata:
            for var in phase2_cols:
                mask_existing = (strata_info == stratum) & (df[:, var] != 0)
                existing_values = df[mask_existing, var]
                missing_mask = (strata_info == stratum) & (df[:, var] == 0)
                num_missing = np.sum(missing_mask)
                resampled_values = np.random.choice(existing_values, size=num_missing, replace=True)
                df[missing_mask, var] = resampled_values
    else:
        for var in phase2_cols:
            mask_existing = df[:, var] != 0
            existing_values = df[mask_existing, var]
            missing_mask = df[:, var] == 0
            num_missing = np.sum(missing_mask)
            resampled_values = np.random.choice(existing_values, size=num_missing, replace=True)
            df[missing_mask, var] = resampled_values

    return df
