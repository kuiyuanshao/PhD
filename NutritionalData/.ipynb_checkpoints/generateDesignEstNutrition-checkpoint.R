pacman::p_load("survey", "stringr")


influenceFunLogistic <- function(modelfit) {
  mm <- model.matrix(modelfit)
  hat <- (t(mm) %*% (mm * modelfit$fitted.values * (1 - modelfit$fitted.values))) / nrow(mm)
  influ <- (mm * resid(modelfit, type = "response")) %*% solve(hat)
  influ
}

calibrateFun <- function(data, truth, design, digit){
  if (design == "/SRS"){
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, NULL), 
                                subset = ~as.logical(R), data = data)
  }else if(design == "/SSRS_exactAlloc"){
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~idx), 
                                subset = ~as.logical(R), data = data)
  }else if(design == "/ODS_exactAlloc"){
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~outcome_strata), 
                                subset = ~as.logical(R), data = data)
  }else if(design == "/RS_exactAlloc"){
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~rs_strata), 
                             subset = ~as.logical(R), data = data)
  }else if(design == "/WRS_exactAlloc"){
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~wrs_strata), 
                             subset = ~as.logical(R), data = data)
  }else if(design == "/SFS_exactAlloc"){
    twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~sfs_strata), 
                                subset = ~as.logical(R), data = data)
  }
  impmodel <- svyglm(c_ln_na_true ~ c_age + c_bmi + c_ln_na_bio1 + high_chol + usborn + female + bkg_pr + bkg_o, 
                     family = gaussian, design = twophase_des)
  data$impute <- as.vector(predict(impmodel, newdata=data, type="response", se.fit=FALSE))
  
  phase1model_imp.1 <- glm(hypertension ~ impute + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, 
                           family = quasibinomial, data = data)
  phase1model_imp.2 <- glm(sbp ~ impute + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, 
                           family = gaussian, data = data)
  
  inffun_imp.1 <- influenceFunLogistic(phase1model_imp.1)
  inffun_imp.2 <- dfbeta(phase1model_imp.2)
  
  colnames(inffun_imp.1)<-paste0("if", 1:ncol(inffun_imp.1))
  colnames(inffun_imp.2)<-paste0("if", 1:ncol(inffun_imp.2))
  
  if (design == "/SRS"){
    twophase_des_imp.1 <- twophase(id = list(~1, ~1), strata = list(NULL, NULL), 
                             subset = ~as.logical(R), data = cbind(data, inffun_imp.1))
    twophase_des_imp.2 <- twophase(id = list(~1, ~1), strata = list(NULL, NULL), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.2))
  }else if(design == "/SSRS_exactAlloc"){
    twophase_des_imp.1 <- twophase(id = list(~1, ~1), strata = list(NULL, ~idx), 
                             subset = ~as.logical(R), data = cbind(data, inffun_imp.1))
    twophase_des_imp.2 <- twophase(id = list(~1, ~1), strata = list(NULL, ~idx), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.2))
  }else if(design == "/RS_exactAlloc"){
    twophase_des_imp.1 <- twophase(id = list(~1, ~1), strata = list(NULL, ~rs_strata), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.1))
    twophase_des_imp.2 <- twophase(id = list(~1, ~1), strata = list(NULL, ~rs_strata), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.2))
  }else if(design == "/WRS_exactAlloc"){
    twophase_des_imp.1 <- twophase(id = list(~1, ~1), strata = list(NULL, ~wrs_strata), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.1))
    twophase_des_imp.2 <- twophase(id = list(~1, ~1), strata = list(NULL, ~wrs_strata), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.2))
  }else if(design == "/ODS_exactAlloc"){
    twophase_des_imp.1 <- twophase(id = list(~1, ~1), strata = list(NULL, ~outcome_strata), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.1))
    twophase_des_imp.2 <- twophase(id = list(~1, ~1), strata = list(NULL, ~outcome_strata), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.2))
  }else{
    twophase_des_imp.1 <- twophase(id = list(~1, ~1), strata = list(NULL, ~sfs_strata), 
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.1))
    twophase_des_imp.2 <- twophase(id = list(~1, ~1), strata = list(NULL, ~sfs_strata),
                                   subset = ~as.logical(R), data = cbind(data, inffun_imp.2))
  }
  califormu.1 <- make.formula(colnames(inffun_imp.1)) 
  califormu.2 <- make.formula(colnames(inffun_imp.2)) 
  
  cali_twophase_imp.1 <- calibrate(twophase_des_imp.1, califormu.1, phase = 2, calfun = "raking")
  cali_twophase_imp.2 <- calibrate(twophase_des_imp.2, califormu.2, phase = 2, calfun = "raking")
  rakingest.1 <- svyglm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, 
                        family = quasibinomial, design = cali_twophase_imp.1)
  rakingest.2 <- svyglm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, 
                        family = gaussian, design = cali_twophase_imp.2)
  ipwest.1 <- svyglm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, 
                     family = quasibinomial, design = twophase_des)
  ipwest.2 <- svyglm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, design = twophase_des)
  
  
  true.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                     female + bkg_o + bkg_pr, family = binomial(), data = truth)
  true.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                     female + bkg_o + bkg_pr, family = gaussian(), data = truth)
  
  
  curr_res.1 <- data.frame(TRUE.Est = coef(true.1),
                           IPW.Est = coef(ipwest.1),
                           RAKING.Est = coef(rakingest.1), 
                           
                           TRUE.Var = diag(vcov(true.1)),
                           IPW.Var = diag(vcov(ipwest.1)),
                           RAKING.Var = diag(vcov(rakingest.1)), 
                           
                           DESIGN = design,
                           DIGIT = digit)
  
  curr_res.2 <- data.frame(TRUE.Est = coef(true.2),
                           IPW.Est = coef(ipwest.2),
                           RAKING.Est = coef(rakingest.2), 
                           
                           TRUE.Var = diag(vcov(true.2)),
                           IPW.Var = diag(vcov(ipwest.2)),
                           RAKING.Var = diag(vcov(rakingest.2)), 
                           
                           DESIGN = design,
                           DIGIT = digit)
  
  return (list(curr_res.1, curr_res.2))
}

setwd("/nesi/project/uoa03789/PhD/SamplingDesigns")
foldernames <- c("/SRS", "/SSRS_exactAlloc", "/RS_exactAlloc", "/WRS_exactAlloc", "/ODS_exactAlloc", "/SFS_exactAlloc")
n <- 100
result_df.1 <- vector("list", n * length(foldernames))
result_df.2 <- vector("list", n * length(foldernames))
m <- 1
for (i in 1:n){
  digit <- str_pad(i, nchar(4444), pad=0)
  print(i)
  for (j in 1:length(foldernames)){
    load(paste0("./NutritionalData/Output/NutritionalData_", digit, ".RData"))
    curr_sample <- read.csv(paste0("./NutritionalData/NutritionalSample", foldernames[j], foldernames[j], "_", digit, ".csv"))
    result <- calibrateFun(curr_sample, pop, foldernames[j], digit)
    result_df.1[[m]] <- result[[1]]
    result_df.2[[m]] <- result[[2]]
    m <- m + 1
  }
}

save(result_df.1, result_df.2, file = "./NutritionalData/NutritionalSample/result_design_based.RData")



