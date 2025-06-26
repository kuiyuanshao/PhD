pacman::p_load(mice, stringr)
source("mice.impute.wnorm.R")
source("mice.impute.cml.R")

generateMiceImpute <- function(data, digit, path, method, weights, strata, split_q, alloc_p, by = "integration", pmm = F){
  inc <- c("c_age", "c_bmi", "c_ln_na_bio1",
           "high_chol", "usborn",
           "female", "bkg_pr", "bkg_o", "sbp", "hypertension")
  pifun <- function(Y, split, prob){
    pi <- ifelse(Y <= split[1], prob[1],
                 ifelse(Y >= split[2], prob[3], prob[2]))
    return (pi)
  }
  pmm_obj <- mice(data, m = 5, print = FALSE,
                  remove.collinear = F, maxit = 1,
                  maxcor = 1.0001, method = method,
                  predictorMatrix = quickpred(data,
                                              include = inc,
                                              exclude = names(data)[!(names(data) %in% inc)]),
                  weights = weights, strata = strata, split_q = split_q, alloc_p = alloc_p,
                  by = by, pifun = pifun,
                  pmm = F)
  
  imputed_data_list <- lapply(1:5, function(i) complete(pmm_obj, i))
  save(imputed_data_list, file = paste0(path, '/MICE_IMPUTE_', digit, '.RData'), compress = 'xz')
}

if(!dir.exists('./Test')){system('mkdir ./Test')}
if(!dir.exists('./Test/Test_rej')){system('mkdir ./Test/Test_rej')}

if(!dir.exists('./Test/Test_rej/pmm')){system('mkdir ./Test/Test_rej/pmm')}
if(!dir.exists('./Test/Test_rej/norm')){system('mkdir ./Test/Test_rej/norm')}
if(!dir.exists('./Test/Test_rej/wnorm')){system('mkdir ./Test/Test_rej/wnorm')}
if(!dir.exists('./Test/Test_rej/cml')){system('mkdir ./Test/Test_rej/cml')}
if(!dir.exists('./Test/Test_rej/cml_rejsamp')){system('mkdir ./Test/Test_rej/cml_rejsamp')}

for (i in 1:100){
  digit <- str_pad(i, nchar(4444), pad=0)
  data <- read.csv(paste0("./Test/ODS_exactAlloc/ODS_exactAlloc_", digit, ".csv"))
  data$c_ln_k_true <- NULL
  data$c_ln_kcal_true <- NULL
  data$c_ln_protein_true <- NULL
  
  data_for_imp <- data
  data_for_imp$W <- NULL
  data_for_imp$outcome_strata <- NULL
  
  weights <- data$W
  strata <- data$outcome_strata
  data$outcome_strata <- NULL
  
  split_q <- quantile(data[["c_ln_na_bio1"]], c(0.19, 0.81))
  alloc_p <- table(data$R, strata)[2,] / colSums(table(data$R, strata))
  curr_path <- './Test/Test_rej/pmm'
  generateMiceImpute(data_for_imp, digit, curr_path, "pmm", weights, strata, split_q, alloc_p)
  
  curr_path <- './Test/Test_rej/norm'
  generateMiceImpute(data_for_imp, digit, curr_path, "norm", weights, strata, split_q, alloc_p)
  
  curr_path <- './Test/Test_rej/wnorm'
  generateMiceImpute(data_for_imp, digit, curr_path, "wnorm", weights, strata, split_q, alloc_p)
  
  curr_path <- './Test/Test_rej/cml'
  generateMiceImpute(data_for_imp, digit, curr_path, "cml", weights, strata, split_q, alloc_p)
  
  curr_path <- './Test/Test_rej/cml_rejsamp'
  generateMiceImpute(data_for_imp, digit, curr_path, "cml", weights, strata, split_q, alloc_p, by = "sampling")

}
