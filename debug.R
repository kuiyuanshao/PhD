pacman::p_load("caret", "mice", "ggplot2", "survival", "data.table", "plyr", "patchwork")

setwd(dir = getwd())
source("./SurvivalData/generateGigantiData.R")
data <- read.csv("./SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/SRS/SRS_0001.csv")
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
true <- generateUnDiscretizeData(alldata)

pmm <- function(yhatobs, yhatmis, yobs, k) {
  idx <- mice::matchindex(d = yhatobs, t = yhatmis, k = k)
  yobs[idx]
}

data_init <- data
R <- data$R
for (i in 1:length(target_variables_1)){
  yobs <- data_init[[target_variables_2[i]]][R == 1]
  yhatmis <- data_init[[target_variables_1[i]]][R == 0]
  yhatobs <- data_init[[target_variables_1[i]]][R == 1]
  data_init[[target_variables_2[i]]][R == 0] <- pmm(yhatobs, yhatmis, yobs, 1)
}
data_init[["ade"]][R == 0] <- data_init[["ade.star"]][R == 1]
#data_init[["ade"]][R == 0] <- sample(data_init[["ade"]][R == 1], 3000, replace = T)

data_init2 <- data
for (i in 1:length(target_variables_1)){
  data_init2[[target_variables_2[i]]][R == 0] <- sample(data_init2[[target_variables_2[i]]][R == 1], 3000, replace = T)
}

R <- data$R
data$R <- NULL
data_init2$R <- NULL


pmm_obj.1 <- mice(data, m = 1, print = FALSE, maxit = 1,
                  maxcor = 1.0001, #ls.meth = "ridge", ridge = 0.1,
                  remove.collinear = F, matchtype = 1L, 
                  visitSequence = c("lastC", "AGE_AT_LAST_VISIT", "last.age",
                                    "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
                                    "ade", "fu"),
                  #predictorMatrix = quickpred(data),
                  method = "pmm")
pmm_res <- complete(pmm_obj.1, 1)
imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = pmm_res, y = FALSE)



predM <- pmm_obj.1$predictorMatrix
visitSeq <- c("lastC", "AGE_AT_LAST_VISIT", "last.age",
              "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
              "ade", "fu")
ry <- R == 1
wy <- R == 0

coef_result <- vector("list", 9)
beta_result <- vector("list", 9)
obs_result <- vector("list", 9)
mis_result <- vector("list", 9)
pmm_result <- vector("list", 9)
for (k in 1:5){
  for (var in 1:length(visitSeq)){
    result <- retrieve(data_init2, visitSeq[var], predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
    data_init2[[visitSeq[var]]][wy] <- result$ypmmmis
    coef_result[[var]] <- rbind(coef_result[[var]], result$coef)
    beta_result[[var]] <- rbind(beta_result[[var]], result$beta)
    obs_result[[var]] <- cbind(obs_result[[var]], result$yhatobs)
    mis_result[[var]] <- cbind(mis_result[[var]], result$yhatmis)
    pmm_result[[var]] <- cbind(pmm_result[[var]], result$ypmmmis)
  }
}

p <- ggplot() 
for (i in 1:5){
  dataf <- data.frame(X = pmm_result[[1]][, i])
  p <- p + 
    geom_density(data = dataf, aes(x = X), colour = i)
}

p

lm(paste0("AGE_AT_LAST_VISIT ~ ", paste0(colnames(coef_result[[2]]), collapse = "+")), data = data)


library(xgboost)
obs.y <- data_init2[!(R == 0), "FirstOImonth"]
obs.data <- data_init2[!(R == 0), -which(names(data) == "FirstOImonth"), drop = FALSE]
mis.data <- data_init2[R == 0, -which(names(data) == "FirstOImonth"), drop = FALSE]

dobs <- xgb.DMatrix(data = as.matrix(obs.data), label = obs.y, nthread = 3)
dmis <- xgb.DMatrix(data = as.matrix(mis.data), nthread = 3)
  
  
obj.type <- "reg:squarederror"
params <- list(max_depth = 6, subsample = 0.7, eta = 0.3)
xgb.fit <- xgb.train(
  data = dobs, objective = obj.type, 
  params = params, nrounds = 30)


library(DiagrammeR)
library(caret)
obs.y <- data_init2[!(R == 0), "FirstOImonth"]
obs.data <- data_init2[!(R == 0), -which(names(data_init2) == "FirstOImonth"), drop = FALSE]
mis.data <- data_init2[R == 0, -which(names(data_init2) == "FirstOImonth"), drop = FALSE]
obs.data <- as.data.frame(obs.data)
train_control <- trainControl(
  method = "none", 
  allowParallel = TRUE
)

tune_grid <- expand.grid(
  nrounds = 30, 
  max_depth = 6,    
  eta = 0.3, 
  gamma = 0,         
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.7 
)
xgb.fit <- train(
  x = obs.data,
  y = obs.y,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid
)

xgb.plot.tree(model = xgb.fit$finalModel, trees = 4)

yhatmis <- predict(xgb.fit, dmis)



# Nutrition:

data <- read.csv("./NutritionalData/NutritionalSample/SRS/SRS_0001.csv")
pmm_obj.1 <- mice(data, m = 1, print = FALSE, maxit = 25,
                  maxcor = 1.0001, #ls.meth = "ridge", ridge = 0.1,
                  remove.collinear = F, matchtype = 1L,
                  #predictorMatrix = quickpred(data),
                  method = "pmm")
pmm_res <- complete(pmm_obj.1, 1)



# Keep putting the random component in, even there is na.
# Assign nas to zeros first.
# Impute the expected values of the truth|measured
### 1. Two observations different calibration predicted values, 
        #different starting values, but pmm will just have one.
# Sample from stratum.
# A Simple Regression. X|X^*

# Often, they will have a set of variables they want, 
# And these can be new information that not observed before (clinical notes).





