pacman::p_load("mice", "mixgb", "dplyr", "caret")

generateMiceImpute <- function(data, digit, path){
  pmm_obj <- mice(data, m = 20, print = FALSE, 
                  remove.collinear = F, maxit = 25,
                  maxcor = 1.0001, 
                  predictorMatrix = quickpred(data))
  
  imputed_data_list <- lapply(1:20, function(i) complete(pmm_obj, i))
  save(imputed_data_list, file = paste0(path, '/MICE_IMPUTE_', digit, '.RData'), compress = 'xz')
}

generateMixgbImpute <- function(data, digit, path, cv_results){
  cleandata <- data_clean(data)
  params <- list(max_depth = 3, subsample = 0.7)
  imputed_data_list <- mixgb(cleandata, m = 20, maxit = 25, nrounds = cv_results$best.nrounds, xgb.params = params)
  save(imputed_data_list, file = paste0(path, '/MIXGB_IMPUTE_', digit, '.RData'), compress = 'xz')
}

source("/nesi/project/uoa03789/PhD/SamplingDesigns/cWGAIN-GP.R")
generateGansImpute <- function(data, digit, path, design, n = 60){
  data$idx <- as.factor(data$idx)
  if (design == "/ODS_exactAlloc"){
    data$outcome_strata <- as.factor(data$outcome_strata)
    onehot_names <- c(paste0("idx.", 1:6), paste0("outcome_strata.", 1:3))
  }else if (design == "/RS_exactAlloc"){
    data$rs_strata <- as.factor(data$rs_strata)
    onehot_names <- c(paste0("idx.", 1:6), paste0("rs_strata.", 1:3))
  }else if (design == "/WRS_exactAlloc"){
    data$wrs_strata <- as.factor(data$wrs_strata)
    onehot_names <- c(paste0("idx.", 1:6), paste0("wrs_strata.", 1:3))
  }else if (design == "/SFS_exactAlloc"){
    data$sfs_strata <- as.factor(data$sfs_strata)
    onehot_names <- c(paste0("idx.", 1:6), paste0("sfs_strata.", 1:3))
  }else{
    onehot_names <- paste0("idx.", 1:6)
  }
  dmy <- dummyVars(" ~ .", data = data)
  trsf <- data.frame(predict(dmy, newdata = data))
  system.time({gain_imp <- cwgangp(trsf, m = 20,
                                   params = list(batch_size = 100, lambda = 10, alpha = 100, n = n, g_layers = 4), 
                                   sampling_info = list(phase1_cols = c("c_ln_na_bio1", "c_ln_k_bio1", 
                                                                        "c_ln_kcal_bio1", "c_ln_protein_bio1"), 
                                                        phase2_cols = c("c_ln_na_true", "c_ln_k_true", 
                                                                        "c_ln_kcal_true", "c_ln_protein_true"), 
                                                        weight_col = "W",
                                                        categorical_cols = c("usborn", "high_chol", "female", "bkg_pr", 
                                                                             "bkg_o", "hypertension", "R", onehot_names)), 
                                   device = "cpu")})
  imputed_data_list <- gain_imp$imputation
  generator_output_list <- gain_imp$sample
  loss <- gain_imp$loss
  save(imputed_data_list, generator_output_list, loss, file = paste0(path, '/GANs_IMPUTE_', digit, '.RData'), compress = 'xz')
}
                              
                              

