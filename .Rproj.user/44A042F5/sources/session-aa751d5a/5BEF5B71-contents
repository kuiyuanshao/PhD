pacman::p_load("caret", "reticulate", "ggplot2", "survival", "data.table", "plyr")

setwd(dir = getwd())
source("./SurvivalData/generateGigantiData.R")
data <- read.csv("./SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/BLS/BLS_0001.csv")
load("./SurvivalData/Output/-1_0_-2_0_-0.25/SurvivalData_0001.RData")

for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "CFAR_PID", "X.1")){
  data[[var]] <- NULL
}

categorical_cols <- c("ade.star", "ade", "R", "fu.star_cut", "X_cut", "Strata")

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
source("./GANs/cWGAIN-GP_MDN.R")

gain_imp <- cwgangp(data, m = 1, 
                    params = list(batch_size = 128, gamma = 5, lambda = 10, alpha = 1, beta = 5, 
                                  lr_g = 5e-5, lr_d = 1e-6, 
                                  n = 5000, g_layers = 4, discriminator_steps = 1), 
                    sampling_info = list(phase1_cols = target_variables_1, 
                                         phase2_cols = target_variables_2, 
                                         weight_col = "W",
                                         categorical_cols = categorical_cols), 
                    device = "cpu",
                    norm_method = "min-max")

imputed_data_list <- gain_imp$imputation
generator_output_list <- gain_imp$sample
loss <- gain_imp$loss

miceimp <- reCalc(imputed_data_list[[1]])
miceimp_exclude <- exclude(miceimp)


imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = miceimp_exclude, y = FALSE)
imp_mod.1



true <- generateUnDiscretizeData(alldata)

ggplot() + 
  geom_density(aes(x = miceimp$AGE_AT_LAST_VISIT), colour = "red") + 
  geom_density(aes(x = true$AGE_AT_LAST_VISIT), colour = "blue") +
  geom_density(aes(x = data$AGE_AT_LAST_VISIT))

ggplot() + 
  geom_density(aes(x = miceimp$fu), colour = "red") + 
  geom_density(aes(x = true$fu), colour = "blue") +
  geom_density(aes(x = data$fu))

ggplot() + 
  geom_density(aes(x = data$ade)) + 
  geom_density(aes(x = miceimp$ade), colour = "red") + 
  geom_density(aes(x = true$ade), colour = "blue")

table(miceimp$ade)


library(patchwork)

all <- NULL
for (i in 1:length(gain_imp$epoch_result)){
  
    k <- gain_imp$epoch_result[[i]][[1]]
    k$epoch <- as.character(i * 100)
    all <- rbind(all, k)
  
}

ggplot(all) +  
  geom_density(aes(x = fu, colour = epoch)) + 
  geom_density(data = data, aes(x = fu), colour = "black") +
  geom_density(data = true, aes(x = fu), colour = "grey")

ggplot(all) +  
  geom_density(aes(x = ade, colour = epoch)) + 
  geom_density(data = data, aes(x = ade), colour = "black") +
  geom_density(data = true, aes(x = ade), colour = "grey")

ggplot(all) +  
  geom_density(aes(x = AGE_AT_LAST_VISIT, colour = epoch)) + 
  geom_density(data = data, aes(x = AGE_AT_LAST_VISIT), colour = "black") +
  geom_density(data = true, aes(x = AGE_AT_LAST_VISIT), colour = "grey")


ggplot(generator_output_list[[1]]) +  
  geom_density(aes(x = AGE_AT_LAST_VISIT), colour = "red") + 
  geom_density(data = data, aes(x = AGE_AT_LAST_VISIT), colour = "black") +
  geom_density(data = true, aes(x = AGE_AT_LAST_VISIT), colour = "grey")

