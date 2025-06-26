import numpy as np
import torch
import torch.nn as nn
from src.diff_models_table import diff_MECSDI
import pandas as pd
import yaml



class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, matched_state = 1, 
                 phase1_cols = 1, phase2_cols = 1):
        super().__init__()
        self.phase1_cols = phase1_cols
        self.phase2_cols = phase2_cols
        self.device = device
        self.target_dim = target_dim
        
        self.emb_feature_dim = config["model"]["featureemb"]
        self.emb_total_dim = self.emb_feature_dim + 1  # for conditional mask
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)
        self.matched_state = matched_state
        
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2
        self.num_steps = config_diff["num_steps"]

        self.diffmodel = diff_MECSDI(config_diff, input_dim, num_steps=self.num_steps, 
                                     matched_state=max(self.matched_state))
        
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"] ** 0.5,
                                    config_diff["beta_end"] ** 0.5,
                                    self.num_steps)** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        alphas_cumprod = np.cumprod(self.alpha_hat)
        self.alpha = alphas_cumprod
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)


    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            if (i in self.phase2_cols):
                sample_ratio = 0.8
                num_observed = observed_mask[i].sum().item()
                num_masked = round(num_observed * sample_ratio)
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask


    def get_side_info(self, cond_mask):
        B, K = cond_mask.shape
        
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).expand(B, K, -1)

        side_info = feature_embed.permute(0, 2, 1) #B, 8, K

        side_mask = cond_mask.unsqueeze(1) #B, 1, K
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info):
        B, K = observed_data.shape
        t = torch.randint(0, max(self.matched_state), [B]).to(self.device)
        
        current_alpha = self.alpha_torch[t]  # (B,1)
        noise = torch.randn_like(observed_data)
        
        phase1rows = (observed_mask[:, self.phase2_cols[0]].reshape(-1) == 0).nonzero()
        phase2rows = observed_mask[:, self.phase2_cols[0]].reshape(-1).nonzero()
        for i in range(len(self.matched_state)):
            alpha_matched = self.alpha_torch[self.matched_state[i]].reshape(1)
            noise_X = (observed_data[:, self.phase1_cols[i]].reshape(-1) - alpha_matched ** 0.5 * observed_data[:, self.phase2_cols[i]].reshape(-1)) / (1.0 - alpha_matched) ** 0.5
            idx = torch.randint(0, B, (B,))
            noise_X = noise_X[idx]
            noise[:, self.phase2_cols[i]] = noise_X
            
        noisy_data = (current_alpha**0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        #residual = (noise - predicted) * weights * target_mask
        #num_eval = (weights * target_mask).sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        
        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K).to(self.device)
        
        for i in range(n_samples):
            current_sample = observed_data
            
            phase1rows = (cond_mask[:, self.phase2_cols[0]].reshape(-1) == 0).nonzero()
            phase2rows = cond_mask[:, self.phase2_cols[0]].reshape(-1).nonzero()
            current_sample[:, self.phase2_cols] = observed_data[:, self.phase1_cols]
            
            for k in range(len(self.matched_state)):
                alpha_matched = self.alpha_torch[self.matched_state[k]].reshape(1)
                if self.matched_state[k] == max(self.matched_state):
                    current_sample[:, self.phase2_cols[k]] = observed_data[:, self.phase1_cols[k]]
                else:
                    noisy_obs = observed_data[:, self.phase1_cols[k]]
                    for j in range(self.matched_state[k], max(self.matched_state)):
                        alpha_matched = self.alpha_torch[self.matched_state[k]].reshape(1)
                        noise = (observed_data[:, self.phase1_cols[k]].reshape(-1) - 
                                     alpha_matched ** 0.5 * observed_data[:, self.phase2_cols[k]].reshape(-1)) / (1.0 - alpha_matched) ** 0.5
                        idx = torch.randint(0, B, (B,))
                        noise = noise[idx]
                        noisy_obs = (self.alpha_hat[j] ** 0.5) * noisy_obs + self.beta[j] ** 0.5 * noise
                    current_sample[:, self.phase2_cols[k]] = noisy_obs
            
            lower_match = [index for index, value in enumerate(self.matched_state) if value != max(self.matched_state)]
            for t in range(max(self.matched_state) - 1, -1, -1):
                if (t + 1) in np.array(self.matched_state)[lower_match]:
                    indices = [index for index, value in enumerate(self.matched_state) if value == (t + 1)]
                    current_sample[:, np.array(self.phase2_cols)[indices]] = observed_data[:, np.array(self.phase1_cols)[indices]]
                    
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
        
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))  # (B,K,L)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data, 
            observed_mask,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        
        if is_train == 0:
            cond_mask = gt_mask
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(cond_mask)

        return self.calc_loss(observed_data, cond_mask, observed_mask, side_info)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(cond_mask)
            
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask


class TabCSDI(CSDI_base):
    def __init__(self, config, device, target_dim = 0, matched_state = 1, 
                 phase1_cols = 1, phase2_cols = 1):
        super(TabCSDI, self).__init__(target_dim, config, device, matched_state, 
                                      phase1_cols, phase2_cols)

    def process_data(self, batch):
        observed_data = batch["observed_data"]
        observed_data = observed_data.to(self.device).float()
        observed_mask = batch["observed_mask"]
        observed_mask = observed_mask.to(self.device).float()
        gt_mask = batch["gt_mask"]
        gt_mask = gt_mask.to(self.device).float()
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask
    
        return (observed_data, observed_mask, gt_mask, for_pattern_mask, cut_length)