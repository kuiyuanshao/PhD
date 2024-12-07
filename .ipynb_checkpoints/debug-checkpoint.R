pacman::p_load("mice", "mixgb", "dplyr", "caret", "survival", "ggplot2", "dplyr")

load("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/Output/-1_0_-2_0_-0.25/SurvivalData_0001.RData")
data <- read.csv("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalSample/-1_0_-2_0_-0.25/SRS/SRS_0001.csv")
source("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/generateGigantiData.R")
target_variables_1 = c("A.star", "D.star", "lastC.star", 
                      "FirstOImonth.star", "FirstARTmonth.star",
                      "AGE_AT_LAST_VISIT.star", "C.star", 
                      "ARTage.star", "OIage.star", "last.age.star", 
                      "ade.star", "fu.star")
target_variables_2 = c("A", "D", "lastC", 
                       "FirstOImonth", "FirstARTmonth",
                       "AGE_AT_LAST_VISIT", "C", 
                       "ARTage", "OIage", "last.age", 
                       "ade", "fu")

for (var in c("A.star", "D.star", "C.star", "A", "D", "C")){
  data[[var]] <- NULL
}

data_true <- generateUnDiscretizeData(alldata)
true_model <- coxph(Surv(fu, ade) ~ X, data = exclude(data_true), y = FALSE)
raw_complete <- coxph(Surv(fu, ade) ~ X, data = exclude(data), y = FALSE)

data$X.1 <- NULL
data$CFAR_PID <- NULL

key_list <- list(
  "lastC" = c("X", "lastC.star"),
  "AGE_AT_LAST_VISIT" = c("AGE_AT_LAST_VISIT.star"),
  "last.age" = c("AGE_AT_LAST_VISIT.star"),
  "FirstARTmonth" = c("FirstOImonth.star", "FirstARTmonth.star", "X", "fu.star"),
  "ARTage" = c("X", "FirstOImonth.star", "fu.star", "ARTage.star"),
  "FirstOImonth" = c("FirstOImonth.star", "FirstARTmonth.star", "X", "ade.star", "fu.star"),
  "OIage" = c("FirstOImonth.star", "FirstARTmonth.star", "X", "ade.star", "fu.star"),
  "ade" = c("X", "OIage.star", "ade.star", "fu.star"),
  "fu" = c("FirstARTmonth.star", "FirstOImonth.star", "X", "AGE_AT_LAST_VISIT", "ade", "fu.star")
)


params <- list(max_depth = 6, subsample = 1, eta = 0.3, gamma = 5, alpha = 10)
all <- list()
for (i in 1:1){
  all[[i]] <- data[, -which(names(data) %in% c("lastC", "AGE_AT_LAST_VISIT", "last.age",
                                               "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
                                               "ade", "fu"))]
}

for (key in c("lastC", "AGE_AT_LAST_VISIT", "last.age",
              "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
              "ade", "fu")){
  data_key <- data[, which(names(data) %in% c(key_list[[key]], key))]
  cv_results <- mixgb_cv(data = data_key, nrounds = 100, xgb.params = params, verbose = FALSE)
  
  imputed_data_list <- mixgb(data_key, m = 1, pmm.type = "auto", maxit = 100, 
                             nrounds = cv_results$best.nrounds, xgb.params = params)
  data[[key]] <- imputed_data_list[[1]][[key]]
  for (i in 1:1){
    all[[i]] <- merge(all[[i]], imputed_data_list[[i]])
  }
}


cv_results.1 <- mixgb_cv(data = cleandat1, nrounds = 100, xgb.params = params, verbose = FALSE)
imputed_data_list.1 <- mixgb(cleandat1, m = 1, pmm.type = "auto", maxit = 100, 
                             nrounds = cv_results.1$best.nrounds, xgb.params = params)

cv_results.2 <- mixgb_cv(data = cleandat2, nrounds = 100, xgb.params = params, verbose = FALSE)
imputed_data_list.2 <- mixgb(cleandat2, m = 1, pmm.type = "auto", maxit = 100, 
                             nrounds = cv_results.2$best.nrounds, xgb.params = params)

imputed_data_list <- list()

for (i in 1:20){
  all <- merge(imputed_data_list.2[[1]], imputed_data_list.1[[1]])
  all$ade <- as.numeric(all$ade) - 1
  imputed_data_list[[i]] <- all
}

all <- merge(imputed_data_list.2[[1]], imputed_data_list.1[[1]])

ggplot(imputed_data_list.1[[1]]) + 
  geom_density(aes(fu), colour = "blue") + 
  geom_density(aes(data_true$fu)) +
  geom_density(aes(data$fu), colour = "red")

ggplot(imputed_data_list.1[[1]]) + 
  geom_density(aes(ade), colour = "blue") + 
  geom_density(aes(data_true$ade)) +
  geom_density(aes(data$ade), colour = "red")


all$ade <- as.numeric(all$ade) - 1
miceimp <- reCalc(all)
miceimp_exclude <- exclude(miceimp)
imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = miceimp_exclude, y = FALSE)
