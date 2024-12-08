pacman::p_load("survey", "readxl", "stringr", "dplyr", "purrr")


setwd(dir = "/nesi/project/uoa03789/PhD/SamplingDesigns")

read_excel_allsheets <- function(filename, tibble = FALSE) {
  sheets <- readxl::excel_sheets(filename)
  x <- lapply(sheets, function(X) readxl::read_excel(filename, sheet = X))
  if(!tibble) x <- lapply(x, as.data.frame)
  names(x) <- sheets
  x
}

find_coef_var <- function(imp, sample = 0, type = 0, design = 0){
  m_coefs.1 <- NULL
  m_coefs.2 <- NULL
  m_vars.1 <- NULL
  m_vars.2 <- NULL
  for (m in 1:length(imp)){
    if (type == "diffusion"){
      ith_imp <- imp[[m]]
      sample_diff <- sample
      phase1inds <- sample_diff$R == 0
      ##--------------Phase-1 Columns [14, 15, 16, 17]--------------
      ##--------------Phase-2 Columns [10, 11, 12, 13]--------------
      cols <- 10:13 + 1
      phase1_cols <- c("c_ln_na_bio1", "c_ln_k_bio1", 
                       "c_ln_kcal_bio1", "c_ln_protein_bio1")
      phase2_cols <- c("c_ln_na_true", "c_ln_k_true", 
                       "c_ln_kcal_true", "c_ln_protein_true")

      for (col in 1:length(cols)){
        #sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * (max(sample_diff[[phase2_cols[col]]], na.rm = T) + 1) - 1
        #sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * 
          #(max(sample_diff[[phase2_cols[col]]], na.rm = T) - min(sample_diff[[phase2_cols[col]]], na.rm = T) + 1) + 
          #(min(sample_diff[[phase2_cols[col]]], na.rm = T) - 1)
        sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * 
          sd(sample_diff[[phase2_cols[col]]], na.rm = T) + 
          mean(sample_diff[[phase2_cols[col]]], na.rm = T)
      }
      
      imp_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, sample_diff, family = binomial())
      imp_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, sample_diff, family = gaussian())
      
    }else{
      ith_imp <- imp[[m]]
      
      imp_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = binomial())
      imp_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = gaussian())
    }
    m_coefs.1 <- rbind(m_coefs.1, coef(imp_mod.1))
    m_coefs.2 <- rbind(m_coefs.2, coef(imp_mod.2))
    m_vars.1 <- rbind(m_vars.1, diag(vcov(imp_mod.1)))
    m_vars.2 <- rbind(m_vars.2, diag(vcov(imp_mod.2)))
  }
  
  var.1 <- 1/20 * colSums(m_vars.1) + (20 + 1) * apply(m_coefs.1, 2, var) / 20
  var.2 <- 1/20 * colSums(m_vars.2) + (20 + 1) * apply(m_coefs.2, 2, var) / 20
  return (list(coef = list(colMeans(m_coefs.1), colMeans(m_coefs.2)), var = list(var.1, var.2)))
}

foldernames <- c("/SRS", "/RS", "/WRS", "/SFS", "/ODS_extTail", "/SSRS_exactAlloc", 
                 "/ODS_exactAlloc", "/RS_exactAlloc", "/WRS_exactAlloc", "/SFS_exactAlloc")
n <- 100
result_df.1 <- vector("list", n * length(foldernames))
result_df.2 <- vector("list", n * length(foldernames))
m <- 1
pb <- txtProgressBar(min = 0, max = n, initial = 0) 
for (i in 1:n){
  setTxtProgressBar(pb, i)
  digit <- str_pad(i, nchar(4444), pad=0)
  for (j in 1:length(foldernames)){
    if (!file.exists(paste0("./Diffusion/imputations", 
                            foldernames[j], foldernames[j], "_", digit, ".xlsx"))){
      next
    }
    cat(digit, ":", foldernames[j], "\n")
    load(paste0("./NutritionalData/Output/NutritionalData_", digit, ".RData"))
    curr_sample <- read.csv(paste0("./NutritionalSample", 
                                   foldernames[j], foldernames[j], "_", digit, ".csv"))
    
    diff_imp <- read_excel_allsheets(paste0("./Diffusion/imputations", 
                                            foldernames[j], foldernames[j], "_", digit, ".xlsx"))
    imp_coefs_vars.diff <- find_coef_var(imp = diff_imp, sample = curr_sample, type = "diffusion", design = foldernames[j])
    
    true.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                       female + bkg_o + bkg_pr, family = binomial(), data = pop)
    true.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                       female + bkg_o + bkg_pr, family = gaussian(), data = pop)
    complete.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                           female + bkg_o + bkg_pr, family = binomial(), data = curr_sample)
    complete.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                           female + bkg_o + bkg_pr, family = gaussian(), data = curr_sample)
    
    load(paste0("./NutritionalData/NutritionalSample/MICE", 
                foldernames[j], "/MICE_IMPUTE_", digit, ".RData"))
    imp_coefs_vars.mice <- find_coef_var(imp = imputed_data_list, sample = curr_sample, type = "mice", design = foldernames[j])
    load(paste0("./NutritionalData/NutritionalSample/MIXGB", 
                foldernames[j], "/MIXGB_IMPUTE_", digit, ".RData"))
    imp_coefs_vars.mixgb <- find_coef_var(imp = imputed_data_list, sample = curr_sample, type = "mixgb", design = foldernames[j])
    load(paste0("./NutritionalData/NutritionalSample/GANs", 
                foldernames[j], "/GANs_IMPUTE_", digit, ".RData"))
    imp_coefs_vars.gans <- find_coef_var(imp = imputed_data_list, sample = curr_sample, type = "gans", design = foldernames[j])

    curr_res.1 <- data.frame(TRUE.Est = coef(true.1),
                             COMPL.Est = coef(complete.1),
                             MICE.imp.Est = imp_coefs_vars.mice$coef[[1]], 
                             MIGXB.imp.Est = imp_coefs_vars.mixgb$coef[[1]], 
                             DIFF.imp.Est = imp_coefs_vars.diff$coef[[1]],
                             GANS.imp.Est = imp_coefs_vars.gans$coef[[1]],
                             
                             TRUE.Var = diag(vcov(true.1)),
                             COMPL.Var = diag(vcov(complete.1)),
                             MICE.imp.Var = imp_coefs_vars.mice$var[[1]], 
                             MIGXB.imp.Var = imp_coefs_vars.mixgb$var[[1]], 
                             DIFF.imp.Var = imp_coefs_vars.diff$var[[1]],
                             GANS.imp.Var = imp_coefs_vars.gans$var[[1]],
                             
                             DESIGN = foldernames[j],
                             DIGIT = digit)
    
    curr_res.2 <- data.frame(TRUE.Est = coef(true.2),
                             COMPL.Est = coef(complete.2),
                             MICE.imp.Est = imp_coefs_vars.mice$coef[[2]], 
                             MIGXB.imp.Est = imp_coefs_vars.mixgb$coef[[2]], 
                             DIFF.imp.Est = imp_coefs_vars.diff$coef[[2]],
                             GANS.imp.Est = imp_coefs_vars.gans$coef[[2]],
                             
                             TRUE.Var = diag(vcov(true.2)),
                             COMPL.Var = diag(vcov(complete.2)),
                             MICE.imp.Var = imp_coefs_vars.mice$var[[2]], 
                             MIGXB.imp.Var = imp_coefs_vars.mixgb$var[[2]], 
                             DIFF.imp.Var = imp_coefs_vars.diff$var[[2]],
                             GANS.imp.Var = imp_coefs_vars.gans$var[[2]],
                             
                             DESIGN = foldernames[j],
                             DIGIT = digit)
    result_df.1[[m]] <- curr_res.1
    result_df.2[[m]] <- curr_res.2
    m <- m + 1
  }
}
close(pb)

save(result_df.1, result_df.2, file = "./NutritionalData/NutritionalSample/result_imputation.RData")


