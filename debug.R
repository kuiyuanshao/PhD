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


pmm_obj.1 <- mice(data, m = 1, print = FALSE, maxit = 5,
                  maxcor = 1.0001, #ls.meth = "ridge", ridge = 0.1,
                  remove.collinear = F, matchtype = 1L, 
                  visitSequence = c("lastC", "AGE_AT_LAST_VISIT", "last.age",
                                    "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
                                    "ade", "fu"),
                  #predictorMatrix = quickpred(data),
                  method = "pmm")

predM <- pmm_obj.1$predictorMatrix
visitSeq <- c("lastC", "AGE_AT_LAST_VISIT", "last.age",
              "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
              "ade", "fu")
ry <- R == 1
wy <- R == 0

result <- retrieve(data_init2, "lastC", predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["lastC"]][wy] <- result$ypmmmis

result <- retrieve(data_init2, "AGE_AT_LAST_VISIT", 
                   predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["AGE_AT_LAST_VISIT"]][wy] <- result$ypmmmis

result <- retrieve(data_init2, "last.age", 
                   predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["last.age"]][wy] <- result$ypmmmis

result <- retrieve(data_init2, "FirstARTmonth", 
                   predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["FirstARTmonth"]][wy] <- result$ypmmmis

result <- retrieve(data_init2, "ARTage", 
                   predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["ARTage"]][wy] <- result$ypmmmis

result <- retrieve(data_init2, "FirstOImonth", 
                   predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["FirstOImonth"]][wy] <- result$ypmmmis

result <- retrieve(data_init2, "OIage", 
                   predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["OIage"]][wy] <- result$ypmmmis

result <- retrieve(data_init2, "ade", 
                   predM, ry, wy, ls.meth = "qr", ridge = 1e-5)
data_init2[["ade"]][wy] <- result$ypmmmis














visitSeq <- c("lastC", "AGE_AT_LAST_VISIT", "last.age",
              "FirstARTmonth", "ARTage", "FirstOImonth", "OIage",
              "ade", "fu")
plot_list <- list()
n_cols <- length(visitSeq)
for (j in 1:5){
  plot_list_iter <- list()
  k <- 1
  for (i in ((j - 1) * n_cols + 1):(j * n_cols)) {
    yhat_data <- data.frame(value = yhatmis_list[, i])
    ymatch_data <- data.frame(value = ymatch_list[, i])
    true_data <- data.frame(value = true[[visitSeq[k]]])
    plot_list_iter[[k]] <- ggplot() +
      geom_density(data = yhat_data, aes(x = value), colour = "red") +
      geom_density(data = ymatch_data, aes(x = value), colour = "blue") +
      geom_density(data = true_data, aes(x = value)) +
      ggtitle(visitSeq[k])
    k <- k + 1
  }
  plot_list[[j]] <- wrap_plots(plot_list_iter, ncol = 3)
}

plot_list[[1]]

