  pacman::p_load(mice, stringr)
  # "norm" >>> linear model.
  # "wnorm" >>> weighted linear model.
  # "norm_wc" >>> linear model + weights as a covariate.
  # "wnorm_x" >>> weighted linear model + covariates X weights interaction.
  # "norm_wx" >>> linear model + covariates X weights interaction.
  # "snorm" >>> individual linear model for each strata.
  
  # "pmm" >>> linear model + pmm imputation.
  # "wpmm" >>> weighted linear model + pmm.
  # "pmm_wc" >>> linear model + weights as a covariate + pmm
  # "wpmm_x" >>> weighted linear model + covariates X weights interation + pmm
  # "pmm_wx" >>> linear model + covariates X weights interaction + pmm
  # "spmm" >>> individual linear model for each strata + pmm
  
  
  # BIAS depends on the stratum. READ sample and population likelihood.
  # We want imputation model in the people who won't sampled.
  # Weights in the unsampled >>> re-weight from the sampled for the unsampled. E[(1/R) / R] = E[(1/R) - 1] = (w - 1)
  
  # Look at the weights, expect value in the unsampled we have.
  # Look at the likelihood. Toy example to see what mice would give as a difference.
  
  
  
  source("mice.impute.wnorm.R")
  source("mice.impute.wnorm_x.R")
  source("mice.impute.norm_wx.R")
  source("mice.impute.snorm.R")
  
  generateMiceImpute <- function(data, digit, path, method, weights, strata, pmm = F){
    pmm_obj <- mice(data, m = 20, print = FALSE, 
                    remove.collinear = F, maxit = 25,
                    maxcor = 1.0001, method = method, 
                    predictorMatrix = quickpred(data), 
                    weights = weights - 1,
                    strata = strata, 
                    pmm = pmm)
    
    imputed_data_list <- lapply(1:20, function(i) complete(pmm_obj, i))
    save(imputed_data_list, file = paste0(path, '/MICE_IMPUTE_', digit, '.RData'), compress = 'xz')
  }
  
  if(!dir.exists('./SFS_exactAlloc')){system('mkdir ./SFS_exactAlloc')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1')){system('mkdir ./SFS_exactAlloc/MICE_w-1')}
  
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/norm')){system('mkdir ./SFS_exactAlloc/MICE_w-1/norm')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/wnorm')){system('mkdir ./SFS_exactAlloc/MICE_w-1/wnorm')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/norm_wc')){system('mkdir ./SFS_exactAlloc/MICE_w-1/norm_wc')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/wnorm_x')){system('mkdir ./SFS_exactAlloc/MICE_w-1/wnorm_x')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/norm_wx')){system('mkdir ./SFS_exactAlloc/MICE_w-1/norm_wx')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/snorm')){system('mkdir ./SFS_exactAlloc/MICE_w-1/snorm')}
  
  
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/pmm')){system('mkdir ./SFS_exactAlloc/MICE_w-1/pmm')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/wpmm')){system('mkdir ./SFS_exactAlloc/MICE_w-1/wpmm')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/pmm_wc')){system('mkdir ./SFS_exactAlloc/MICE_w-1/pmm_wc')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/wpmm_x')){system('mkdir ./SFS_exactAlloc/MICE_w-1/wpmm_x')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/pmm_wx')){system('mkdir ./SFS_exactAlloc/MICE_w-1/pmm_wx')}
  if(!dir.exists('./SFS_exactAlloc/MICE_w-1/spmm')){system('mkdir ./SFS_exactAlloc/MICE_w-1/spmm')}
  
  for (i in 1:100){
    digit <- str_pad(i, nchar(4444), pad=0)
    data <- read.csv(paste0("../NutritionalData/NutritionalSample/SFS_exactAlloc/SFS_exactAlloc_", digit, ".csv"))
    
    data_wow <- data
    data_wow$W <- NULL
    data_wow$sfs_strata <- NULL
    
    weights <- data$W
    strata <- data$sfs_strata
    
    data$sfs_strata <- NULL
    
    ### Linear wow PMM
    
    #imputation model excludes weights and strata info
    curr_path <- './SFS_exactAlloc/MICE_w-1/norm'
    generateMiceImpute(data_wow, digit, curr_path, "norm", weights, strata)
    
    #weighted imputation model
    curr_path <- './SFS_exactAlloc/MICE_w-1/wnorm'
    generateMiceImpute(data_wow, digit, curr_path, "wnorm", weights, strata)
    
    #imputation model that uses weights as a covariate
    curr_path <- './SFS_exactAlloc/MICE_w-1/norm_wc'
    generateMiceImpute(data, digit, curr_path, "norm", weights, strata)
    
    #weighted imputation model and additionally uses weights interacting with other covariates. 
    curr_path <- './SFS_exactAlloc/MICE_w-1/wnorm_x'
    generateMiceImpute(data_wow, digit, curr_path, "wnorm_x", weights, strata)
    
    #imputation model that uses weights interacting with other covariates. 
    curr_path <- './SFS_exactAlloc/MICE_w-1/norm_wx'
    generateMiceImpute(data_wow, digit, curr_path, "norm_wx", weights, strata)
    
    #individual imputation model for each strata
    curr_path <- './SFS_exactAlloc/MICE_w-1/snorm'
    generateMiceImpute(data_wow, digit, curr_path, "snorm", weights, strata)
    
    
    ### PMM
    
    #imputation model excludes weights and strata info
    curr_path <- './SFS_exactAlloc/MICE_w-1/pmm'
    generateMiceImpute(data_wow, digit, curr_path, "pmm", weights, strata, pmm = T)
    
    #weighted imputation model
    curr_path <- './SFS_exactAlloc/MICE_w-1/wpmm'
    generateMiceImpute(data_wow, digit, curr_path, "wnorm", weights, strata, pmm = T)
    
    #imputation model that uses weights as a covariate
    curr_path <- './SFS_exactAlloc/MICE_w-1/pmm_wc'
    generateMiceImpute(data, digit, curr_path, "pmm", weights, strata, pmm = T)
    
    #weighted imputation model and additionally uses weights interacting with other covariates. 
    curr_path <- './SFS_exactAlloc/MICE_w-1/wpmm_x'
    generateMiceImpute(data_wow, digit, curr_path, "wnorm_x", weights, strata, pmm = T)
    
    #imputation model that uses weights interacting with other covariates. 
    curr_path <- './SFS_exactAlloc/MICE_w-1/pmm_wx'
    generateMiceImpute(data_wow, digit, curr_path, "norm_wx", weights, strata, pmm = T)
    
    #individual imputation model for each strata
    curr_path <- './SFS_exactAlloc/MICE_w-1/spmm'
    generateMiceImpute(data_wow, digit, curr_path, "snorm", weights, strata, pmm = T)
  }
  
  
