pacman::p_load("caret", "reticulate", "ggplot2", "survival", "data.table", "plyr")

source("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/generateGigantiData.R")
data <- read.csv("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/BLS/BLS_0001.csv")
load("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/Output/-1_0_-2_0_-0.25/SurvivalData_0001.RData")

for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "CFAR_PID", "X.1")){
  data[[var]] <- NULL
}

data$X_cut <- as.factor(data$X_cut)
data$fu.star_cut <- as.factor(data$X_cut)
data$Strata <- as.factor(data$Strata)
dmy <- dummyVars(" ~ .", data = data)
data <- data.frame(predict(dmy, newdata = data))
categorical_cols <- c("ade.star", "ade", "R", paste0("fu.star_cut.", 1:2), paste0("X_cut.", 1:2), paste0("Strata.", 1:8))

fu_maxoffset <- (data$fu + 1) / (max(data$fu, na.rm = T) + 1)
min_max <- 2* (data$fu - min(data$fu, na.rm = T) + 1e-6) / (max(data$fu, na.rm = T) - min(data$fu, na.rm = T) + 1e-6) - 1
fu_log <- (log(data$fu + 1) - min(log(data$fu + 1), na.rm = T) + 1e-6) / (max(log(data$fu + 1), na.rm = T) - min(log(data$fu + 1), na.rm = T) + 1e-6)
standard_norm <- (data$fu - mean(data$fu, na.rm = T) + 1e-6) / (sd(data$fu, na.rm = T) + 1e-6)

ggplot() + 
  #geom_density(aes(x = data$fu)) + 
  geom_density(aes(x = fu_maxoffset), colour = "red") + 
  geom_density(aes(x = min_max), colour = "blue") + 
  geom_density(aes(x = fu_log), colour = "black") + 
  geom_density(aes(x = standard_norm), colour = "purple")



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
source("/nesi/project/uoa03789/PhD/SamplingDesigns/GANs/cWGAIN-GP.R")
gain_imp <- cwgangp(data, m = 20, 
                    params = list(batch_size = 128, gamma = 1, lambda = 10, alpha = 1, beta = 0.5, 
                                  lr_g = 1e-4, lr_d = 1e-6, 
                                  n = 3000, g_layers = 4, discriminator_steps = 1), 
                    sampling_info = list(phase1_cols = target_variables_1, 
                                         phase2_cols = target_variables_2, 
                                         weight_col = "W",
                                         categorical_cols = categorical_cols), 
                    device = "cpu")

imputed_data_list <- gain_imp$imputation
generator_output_list <- gain_imp$sample


miceimp <- reCalc(imputed_data_list[[1]])
miceimp_exclude <- exclude(miceimp)

imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = miceimp_exclude, y = FALSE)
imp_mod.1

true <- generateUnDiscretizeData(alldata)
ggplot() + 
  geom_density(aes(x = miceimp$fu), colour = "red") + 
  geom_density(aes(x = true$fu), colour = "blue") +
  geom_density(aes(x = data$fu))

ggplot() + 
  geom_density(aes(x = data$ade)) + 
  geom_density(aes(x = miceimp$ade), colour = "red") + 
  geom_density(aes(x = true$ade), colour = "blue")

library(patchwork)

all <- NULL
for (i in 1:length(gain_imp$epoch_result)){
  k <- gain_imp$epoch_result[[i]][[1]]
  k$epoch <- as.character(i * 100)
  all <- rbind(all, k)
}

ggplot(all) +  
  geom_density(aes(x = fu, colour = epoch)) + 
  geom_density(data = data, aes(x = fu), colour = "black")+
  geom_density(data = true, aes(x = fu), colour = "grey")


ggplot(generator_output_list[[1]]) +  
  geom_point(aes(x = fu.star, y = fu))
ggplot(generator_output_list[[1]]) +  
  geom_point(aes(x = X, y = fu)) #+ 
  #facet_wrap(~epoch)
