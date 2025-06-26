pacman::p_load("caret", "mice", "ggplot2", "survival", "data.table", "plyr", "patchwork")
reCalc <- function(data){
                          # Bound FirstOImonth and FirstARTmonth between 0 and 101
  data <- data %>% mutate(FirstOImonth = round(ifelse(FirstOImonth > 101, 101, FirstOImonth)),
                          FirstARTmonth = round(ifelse(FirstARTmonth > 101, 101, FirstARTmonth)),
                          # Bound AGE_AT_LAST_VISIT between 0 and 100 (adjusted for month-to-year conversion)
                          AGE_AT_LAST_VISIT = ifelse(
                            (AGE_AT_LAST_VISIT / (30.437 / 365.25)) >= 100, 
                            100 * (30.437 / 365.25), AGE_AT_LAST_VISIT
                          ),
                          # Bound last.age between 0 and 101
                          last.age = ifelse(
                            (last.age / (30.437 / 365.25)) >= 101, 
                            101 * (30.437 / 365.25), last.age
                          ),
                          # Bound OIage between 0 and 101
                          OIage = ifelse(
                            (OIage / (30.437 / 365.25)) > 101, 
                            101 * (30.437 / 365.25), OIage
                          ),
                          # Bound ARTage between 0 and 101
                          ARTage = ifelse(
                            (ARTage / (30.437 / 365.25)) > 101, 
                            101 * (30.437 / 365.25), ARTage
                          ),
                          ade = pmin(pmax(round(ade), 0), 1),
                          fu = ifelse(fu < 0, 0, fu))
  return (data)
}

exclude <- function(data, 
                    FirstARTmonth = "FirstARTmonth", 
                    OIage = "OIage", ARTage = "ARTage", 
                    fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT"){
  data <- as.data.frame(data)
  data$exclude.no.art <- ifelse(data[[FirstARTmonth]] >= 101, 1, ifelse(data[[ARTage]] > data[[AGE_AT_LAST_VISIT]], 1, 0))
  data$exclude.prior.ade <- ifelse(data[[OIage]] != (101 * 30.437/365.25) & data[[FirstARTmonth]] != 101 & data[[OIage]] < data[[ARTage]], 1, 0)
  data$exclude.not.naive <- ifelse(data[[FirstARTmonth]] != 101 & data[[ARTage]] < 0, 1, 0)
  data$exclude <- with(data, ifelse(exclude.no.art==1 | exclude.prior.ade==1 | exclude.not.naive, 1, 0))
  
  data <- data[data$exclude == 0, ]
  data <- data[data[[fu]] > 0, ]
  
  return (data)
}
setwd(dir = getwd())
source("../SurvivalData/generateGigantiData.R")
data <- read.csv("../SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/SRS/SRS_0001.csv")
load("../SurvivalData/Output/-1_0_-2_0_-0.25/SurvivalData_0001.RData")

for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "CFAR_PID", "X.1")){
  data[[var]] <- NULL
}

vars <- c("lastC", 
          "FirstOImonth", "FirstARTmonth",
          "AGE_AT_LAST_VISIT",
          "ARTage", "OIage", "last.age", "fu")

categorical_cols <- c("ade.star", "ade", "R")
target_variables_1 <- c("lastC.star", 
                        "FirstOImonth.star", "FirstARTmonth.star",
                        "AGE_AT_LAST_VISIT.star", 
                        "ARTage.star", "OIage.star", "last.age.star", 
                        "ade.star", "fu.star")
target_variables_2 <- c("lastC", 
                        "FirstOImonth", "FirstARTmonth",
                        "AGE_AT_LAST_VISIT",
                        "ARTage", "OIage", "last.age", 
                        "ade", "fu")
data_info = list(phase1_vars = target_variables_1, 
                 phase2_vars = target_variables_2, 
                 weight_var = "W",
                 cat_vars = categorical_cols,
                 num_vars = names(data)[!names(data) %in% c("W", categorical_cols)])

source("mmer.impute.cwgangp.R") 
imp_obj <- mmer.impute.cwgangp(data, m = 1, 
                               data_info = data_info, epochs = 10000, save.step = 1000)

imputed_data_list <- imp_obj$imputation
generator_output_list <- imp_obj$gsample
loss <- imp_obj$loss

replicate <- reCalc(imp_obj$step_result[[6]][[1]])
st1 <- exclude(replicate, FirstARTmonth = "FirstARTmonth", 
               OIage = "OIage", ARTage = "ARTage", 
               fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT")
imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = st1, y = FALSE)
imp_mod.1
truth <- generateUnDiscretizeData(alldata)
truthexc <- exclude(truth)
true <- coxph(Surv(fu, ade) ~ X, data = truthexc, y = FALSE)

ggplot() + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`G Loss`, colour = "red")) + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`D Loss`, colour = "blue")) + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`MSE`, colour = "orange")) + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`Cross-Entropy`, colour = "purple"))

ggplot() + geom_density(aes(x = imputed_data_list[[1]]$fu)) + 
  geom_density(aes(x = truth$fu))

ggplot() + geom_density(aes(x = imputed_data_list[[1]]$ade)) + 
  geom_density(aes(x = truth$ade)) 

