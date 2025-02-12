pacman::p_load(mice, stringr)
# "wnorm" >>> weighted imputation model.
source("mice.impute.wnorm.R")
# "wnorm_x" >>> weighted imputation model + covariates X weights interaction.
source("mice.impute.wnorm_x.R")
# "norm_wx" >>> imputation model + covairates X weights interaction.
source("mice.impute.norm_wx.R")
# "snorm" >>> individual imputation model for each strata.
source("mice.impute.snorm.R")

generateMiceImpute <- function(data, digit, path, method, weights, strata){
  pmm_obj <- mice(data, m = 20, print = FALSE, 
                  remove.collinear = F, maxit = 25,
                  maxcor = 1.0001, method = method, 
                  predictorMatrix = quickpred(data), 
                  weights = weights,
                  strata = strata)
  
  imputed_data_list <- lapply(1:20, function(i) complete(pmm_obj, i))
  save(imputed_data_list, file = paste0(path, '/MICE_IMPUTE_', digit, '.RData'), compress = 'xz')
}

if(!dir.exists('./SFS_exactAlloc')){system('mkdir ./SFS_exactAlloc')}
if(!dir.exists('./SFS_exactAlloc/MICE')){system('mkdir ./SFS_exactAlloc/MICE')}

if(!dir.exists('./SFS_exactAlloc/MICE/norm')){system('mkdir ./SFS_exactAlloc/MICE/norm')}
if(!dir.exists('./SFS_exactAlloc/MICE/wnorm')){system('mkdir ./SFS_exactAlloc/MICE/wnorm')}
if(!dir.exists('./SFS_exactAlloc/MICE/norm_wc')){system('mkdir ./SFS_exactAlloc/MICE/norm_wc')}
if(!dir.exists('./SFS_exactAlloc/MICE/wnorm_x')){system('mkdir ./SFS_exactAlloc/MICE/wnorm_x')}
if(!dir.exists('./SFS_exactAlloc/MICE/norm_wx')){system('mkdir ./SFS_exactAlloc/MICE/norm_wx')}
if(!dir.exists('./SFS_exactAlloc/MICE/snorm')){system('mkdir ./SFS_exactAlloc/MICE/snorm')}

for (i in 1:100){
  digit <- str_pad(i, nchar(4444), pad=0)
  data <- read.csv(paste0("../NutritionalData/NutritionalSample/SFS_exactAlloc/SFS_exactAlloc_", digit, ".csv"))
  
  data_wow <- data
  data_wow$W <- NULL
  data_wow$sfs_strata <- NULL
  
  weights <- data$W
  strata <- data$sfs_strata
  
  data$sfs_strata <- NULL
  
  #imputation model excludes weights and strata info
  curr_path <- './SFS_exactAlloc/MICE/norm'
  generateMiceImpute(data_wow, digit, curr_path, "norm", weights, strata)
  
  #weighted imputation model
  curr_path <- './SFS_exactAlloc/MICE/wnorm'
  generateMiceImpute(data_wow, digit, curr_path, "wnorm", weights, strata)
  
  #imputation model that uses weights as a covariate
  curr_path <- './SFS_exactAlloc/MICE/norm_wc'
  generateMiceImpute(data, digit, curr_path, "norm", weights, strata)
  
  #weighted imputation model and additionally uses weights interacting with other covariates. 
  curr_path <- './SFS_exactAlloc/MICE/wnorm_x'
  generateMiceImpute(data_wow, digit, curr_path, "wnorm_x", weights, strata)
  
  #imputation model that uses weights interacting with other covariates. 
  curr_path <- './SFS_exactAlloc/MICE/norm_wx'
  generateMiceImpute(data_wow, digit, curr_path, "wnorm_x", weights, strata)
  
  #individual imputation model for each strata
  curr_path <- './SFS_exactAlloc/MICE/snorm'
  generateMiceImpute(data_wow, digit, curr_path, "snorm", weights, strata)
}


