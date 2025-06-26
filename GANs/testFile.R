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
truth <- generateUnDiscretizeData(alldata)
population <- truth
missing_ratio <- 0.75
n_subject <- length(unique(population[["CFAR_PID"]]))
n_phase2 <- as.integer(n_subject * (1 - missing_ratio))
population <- truth
data_bls <- population
quantile_split <- c(0.5)
fu.star_cut <- ifelse(population$FirstARTmonth.star > 29, 1, 0)
X_cut <- cut(population[["OIage.star"]], breaks = c(-Inf, quantile(population[["OIage.star"]], probs = quantile_split), Inf), 
             labels = paste(1:(length(quantile_split) + 1), sep=','))
data_bls$fu.star_cut <- fu.star_cut
data_bls$X_cut <- X_cut
strata_indicators <- model.matrix(object = ~ -1 + ade.star + fu.star_cut + X_cut,
                                  data = data_bls)
vars_to_balance  <- diag(rep(1 - missing_ratio, times = n_subject)) %*% strata_indicators
id_phase2 <- samplecube(X = vars_to_balance, pik = rep(1 - missing_ratio, times = n_subject), comment = F, method = 1)
selected_sample_bls <- getdata(data_bls, id_phase2)
n_h <- selected_sample_bls %>%
  dplyr::group_by(ade.star, fu.star_cut, X_cut) %>%
  dplyr::summarize(n_h = n())
N_h <- data_bls %>%
  dplyr::group_by(ade.star, fu.star_cut, X_cut) %>%
  dplyr::summarize(N_h = n())
N_h$W <- N_h$N_h / n_h$n_h
N_h$Strata <- as.numeric(as.factor(N_h$W))
data_bls <- merge(data_bls, N_h)
data_bls <- data_bls %>%
  dplyr::mutate(R = id_phase2, dplyr::across(dplyr::all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))

for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "CFAR_PID", "X.1", "N_h", "X_cut", "fu.star_cut", "Strata")){
  data_bls[[var]] <- NULL
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
                 num_vars = names(data_bls)[!names(data_bls) %in% c("W", categorical_cols)])

source("mmer.impute.cwgangp.R") 
imp_obj <- mmer.impute.cwgangp(data_bls, m = 1, 
                               params = list(batch_size = 500, n_g_layers = 5, n_d_layers = 3), 
                               data_info = data_info, epochs = 2000, save.step = 1000)
#save(imp_obj, file = "imp_obj.RData")
#load("imp_obj.RData")
imputed_data_list <- imp_obj$imputation
generator_output_list <- imp_obj$gsample
loss <- imp_obj$loss

imputation <- imputed_data_list[[1]]
gsample <- generator_output_list[[1]]

replicate <- reCalc(imputation)

st1 <- exclude(replicate, FirstARTmonth = "FirstARTmonth", 
               OIage = "OIage", ARTage = "ARTage", 
               fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT")
imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = st1, y = FALSE)
imp_mod.1

truthexc <- exclude(truth)
true <- coxph(Surv(fu, ade) ~ X, data = truthexc, y = FALSE)
true
ggplot() + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`G Loss`), colour = "red") + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`D Loss`), colour = "blue") + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`MSE`), colour = "orange") + 
  geom_line(aes(x = 1:nrow(loss), y = loss$`Cross-Entropy`), colour = "purple")

ggplot() + geom_density(aes(x = imputation$fu), colour = "red") + 
  geom_density(aes(x = truth$fu), colour = "blue") +
  geom_density(aes(x = data_bls$fu))

ggplot() + geom_density(aes(x = imputation$ade), colour = "red") + 
  geom_density(aes(x = truth$ade), colour = "blue")  +
  geom_density(aes(x = data$ade))


ggplot() + geom_point(aes(x = X, y = fu), data = data)

data1 <- read.csv("../SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/SRS/SRS_0001.csv")
data2 <- read.csv("../SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/BLS/BLS_0001.csv")

ggplot() + geom_density(aes(ade), data = data1, colour = "blue") + 
  geom_density(aes(ade), data = data2, colour = "red")
summary(data1$FirstARTmonth)
summary(data2$FirstARTmonth)

target_variables_2 <- c("lastC", 
                        "FirstOImonth", "FirstARTmonth",
                        "AGE_AT_LAST_VISIT",
                        "ARTage", "OIage", "last.age", 
                        "ade", "fu")

coxph(Surv(fu, ade) ~ X, data = data1, y = FALSE)
coxph(Surv(fu, ade) ~ X, data = data2, y = FALSE)

lm(fu ~ X, data = data1)
lm(fu ~ X, data = data2)
table(data2$X_cut, data2$fu.star_cut, data2$ade.star)
