pacman::p_load("mice", "mixgb", "dplyr", "caret")
setwd(dir = "/nesi/project/uoa03789/PhD/SamplingDesigns")


generateMiceImpute <- function(data, digit, path, type){
  if (type == "survival"){
    for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "X.1", "CFAR_PID")){
      data[[var]] <- NULL
    }
    
    predMat <- quickpred(data)
    #lastC wants: c("lastC.star")
    predMat["lastC", ] <- colnames(predMat) %in% c("X", "lastC.star")
    #AGE_AT_LAST_VISIT wants: c("X", "AGE_AT_LAST_VISIT.star")
    predMat["AGE_AT_LAST_VISIT", ] <- colnames(predMat) %in% c("AGE_AT_LAST_VISIT.star")
    #last.age wants: c("AGE_AT_LAST_VISIT")
    predMat["last.age", ] <- colnames(predMat) %in% c("AGE_AT_LAST_VISIT.star")
    #FirstARTmonth wants: c("FirstOImonth.star", "FirstARTmonth.star", "X", "fu.star")
    predMat["FirstARTmonth", ] <- colnames(predMat) %in% c("FirstOImonth.star", "FirstARTmonth.star", "X", "fu.star")
    #ARTage wants: c("X", "FirstOImonth.star", "fu.star", "ARTage.star")
    predMat["ARTage", ] <- colnames(predMat) %in% c("X", "FirstOImonth.star", "fu.star", "ARTage.star")
    #FirstOImonth wants: c("FirstOImonth.star", "FirstARTmonth.star", "X", "ade.star", "fu.star")
    predMat["FirstOImonth", ] <- colnames(predMat) %in% c("FirstOImonth.star", "FirstARTmonth.star", "X", "ade.star", "fu.star")
    #OIage wants: c("FirstOImonth.star", "FirstARTmonth.star", "X", "ade.star", "fu.star")
    predMat["OIage", ] <- colnames(predMat) %in% c("FirstOImonth.star", "FirstARTmonth.star", "X", "ade.star", "fu.star")
    #ade wants: c("X", "OIage.star", "ade.star", "fu.star") #Do
    predMat["ade", ] <- colnames(predMat) %in% c("X", "OIage.star", "ade.star", "fu.star")
    #fu wants: c("FirstARTmonth.star", "FirstOImonth.star", "X", "AGE_AT_LAST_VISIT", "ade.star", "fu.star") 
    #use imputed ade insteadof ade.star since a binary imputed state tends to be stable, and ade addressed bias in the ade.star
    predMat["fu", ] <- colnames(predMat) %in% c("FirstARTmonth.star", "FirstOImonth.star", "X", "AGE_AT_LAST_VISIT", "ade", "fu.star")
    method <- rep("pmm", ncol(data))
    
    # If using predictive mean matching in the exclusion criteria related variables, there will be too many observations excluded
    
    method[match(c("lastC", "AGE_AT_LAST_VISIT", "last.age",
                   "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
                   "ade", "fu"), names(data))] <- c("norm", "norm", "norm", 
                                                    "norm", "norm",
                                                    "norm", "norm",
                                                    "pmm", "pmm")
    
    pmm_obj <- mice(data, m = 20, print = FALSE, 
                    remove.collinear = F, maxit = 50,
                    maxcor = 1.0001,
                    visitSequence = c("lastC", "AGE_AT_LAST_VISIT", "last.age",
                                      "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
                                      "ade", "fu"),
                    predictorMatrix = predMat,
                    method = method)
    imputed_data_list <- lapply(1:20, function(i) complete(pmm_obj, i))
    save(imputed_data_list, file = paste0(path, '/MICE_IMPUTE_', digit, '.RData'), compress = 'xz')
  }else{
    pmm_obj <- mice(data, m = 20, print = FALSE, 
                    remove.collinear = F, maxit = 25,
                    maxcor = 1.0001, 
                    predictorMatrix = quickpred(data))
    
    imputed_data_list <- lapply(1:20, function(i) complete(pmm_obj, i))
    save(imputed_data_list, file = paste0(path, '/MICE_IMPUTE_', digit, '.RData'), compress = 'xz')
  }
}

generateMixgbImpute <- function(data, digit, path, type){
  if (type == "survival"){
    source("./mixgbME.R")
    for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "X.1", "CFAR_PID")){
      data[[var]] <- NULL
    }
    params <- list(max_depth = 6, subsample = 0.7, eta = 0.3)
    cleandata <- data_clean(data)
    cv_results <- mixgb_cv(data = cleandata, nrounds = 100, xgb.params = params, verbose = FALSE)
    imputed_data_list <- mixgb(cleandata, m = 20, pmm.type = "auto", maxit = 50, 
                               nrounds = cv_results$best.nrounds, xgb.params = params, verbose = F)
    save(imputed_data_list, cv_results, file = paste0(path, '/MIXGB_IMPUTE_', digit, '.RData'), compress = 'xz')
  }else{
    cleandata <- data_clean(data)
    params <- list(max_depth = 3, subsample = 0.7)
    cv_results <- mixgb_cv(data = cleandata, nrounds = 100, xgb.params = params, verbose = FALSE)
    imputed_data_list <- mixgb(cleandata, m = 20, maxit = 25, nrounds = cv_results$best.nrounds, xgb.params = params)
    save(imputed_data_list, cv_results, file = paste0(path, '/MIXGB_IMPUTE_', digit, '.RData'), compress = 'xz')
  }
}

generateGansImpute <- function(data, digit, path, design, type){
  source("./GANs/cWGAIN-GP.R")
  if (type == "nutrition"){
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
                                     params = list(batch_size = 128, gamma = 1, lambda = 1, alpha = 1, beta = 1,
                                                   lr_g = 5e-5, lr_d = 1e-6,
                                                   n = 3000, g_layers = 4, discriminator_steps = 1), 
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
  }else{
    for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "X.1", "CFAR_PID")){
      data[[var]] <- NULL
    }
    
    if (design == "/SRS"){
      categorical_cols <- c("ade.star", "ade", "R")
    }else if (design == "/BLS"){
      data$X_cut <- as.factor(data$X_cut)
      data$fu.star_cut <- as.factor(data$X_cut)
      data$Strata <- as.factor(data$Strata)
      dmy <- dummyVars(" ~ .", data = data)
      data <- data.frame(predict(dmy, newdata = data))
      categorical_cols <- c("ade.star", "ade", "R", paste0("fu.star_cut.", 1:2), paste0("X_cut.", 1:2), paste0("Strata.", 1:8))
    }
    
    
    target_variables_1 = c("lastC.star", 
                           "FirstOImonth.star", "FirstARTmonth.star",
                           "AGE_AT_LAST_VISIT.star", 
                           "ARTage.star", "OIage.star", "last.age.star", 
                           "ade.star", "fu.star")
    target_variables_2 = c("lastC", 
                           "FirstOImonth", "FirstARTmonth",
                           "AGE_AT_LAST_VISIT",
                           "ARTage", "OIage", "last.age", 
                           "ade", "fu")
    
    gain_imp <- cwgangp(data, m = 20, 
                        params = list(batch_size = 256, gamma = 1, lambda = 1, alpha = 1, beta = 0.5, 
                                      lr_g = 1e-4, lr_d = 1e-6, 
                                      n = 10000, g_layers = 6, discriminator_steps = 1), 
                        sampling_info = list(phase1_cols = target_variables_1, 
                                             phase2_cols = target_variables_2, 
                                             weight_col = "W",
                                             categorical_cols = categorical_cols), 
                        device = "cpu")
    imputed_data_list <- gain_imp$imputation
    generator_output_list <- gain_imp$sample
    loss <- gain_imp$loss
    save(imputed_data_list, generator_output_list, loss, file = paste0(path, '/GANs_IMPUTE_', digit, '.RData'), compress = 'xz')
    
    miceimp <- reCalc(imputed_data_list[[1]])
    miceimp_exclude <- exclude(miceimp)
    
    imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = miceimp_exclude, y = FALSE)
    imp_mod.1
  }
}


generateCycleGansImpute <- function(data, digit, path, design, type){
  source("./GANs/ccycleWGAN-GP.R")
  if (type == "nutrition"){
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
    system.time({gain_imp <- cyclewgangp(trsf, m = 20,
                                         params = list(batch_size = 128, gamma = 1, lambda = 1, alpha = 1, beta = 1, 
                                                       lr_g = 5e-5, lr_d = 1e-6,
                                                       n = 3500, g_layers = 4, discriminator_steps = 1), 
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
    save(imputed_data_list, generator_output_list, loss, file = paste0(path, '/cycleGANs_IMPUTE_', digit, '.RData'), compress = 'xz')
  }else{
    for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "X.1", "CFAR_PID")){
      data[[var]] <- NULL
    }
    
    if (design == "/SRS"){
      categorical_cols <- c("ade.star", "ade", "R")
    }else if (design == "/BLS"){
      data$X_cut <- as.factor(data$X_cut)
      data$fu.star_cut <- as.factor(data$X_cut)
      data$Strata <- as.factor(data$Strata)
      dmy <- dummyVars(" ~ .", data = data)
      data <- data.frame(predict(dmy, newdata = data))
      categorical_cols <- c("ade.star", "ade", "R", paste0("fu.star_cut.", 1:2), paste0("X_cut.", 1:2), paste0("Strata.", 1:8))
    }
    
    
    target_variables_1 = c("lastC.star", 
                           "FirstOImonth.star", "FirstARTmonth.star",
                           "AGE_AT_LAST_VISIT.star", 
                           "ARTage.star", "OIage.star", "last.age.star", 
                           "ade.star", "fu.star")
    target_variables_2 = c("lastC", 
                           "FirstOImonth", "FirstARTmonth",
                           "AGE_AT_LAST_VISIT",
                           "ARTage", "OIage", "last.age", 
                           "ade", "fu")
    gain_imp <- cyclewgangp(data, m = 20, 
                            params = list(batch_size = 128, gamma = 1, lambda = 1, alpha = 1, beta = 1, 
                                          lr_g = 5e-5, lr_d = 1e-6,
                                      n = 5000, g_layers = 6, discriminator_steps = 1), 
                            sampling_info = list(phase1_cols = target_variables_1, 
                                             phase2_cols = target_variables_2, 
                                             weight_col = "W",
                                             categorical_cols = categorical_cols), 
                            device = "cpu")
    imputed_data_list <- gain_imp$imputation
    generator_output_list <- gain_imp$sample
    loss <- gain_imp$loss
    save(imputed_data_list, generator_output_list, loss, file = paste0(path, '/cycleGANs_IMPUTE_', digit, '.RData'), compress = 'xz')
    
  }
}