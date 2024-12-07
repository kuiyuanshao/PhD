pacman::p_load("stringr")

args <- commandArgs(trailingOnly = TRUE)
task_id <- as.numeric(args[1])      # SLURM_ARRAY_TASK_ID

# Survival Data Imputation
foldernames <- c("/-1_0_-2_0_-0.25", "/-1_0_-2_0_0", "/-1_0_-2_0_0.25",
                 "/-1_0.25_-2_0.5_-0.25", "/-1_0.25_-2_0.5_0", "/-1_0.25_-2_0.5_0.25",
                 "/-1_0.5_-2_1_-0.25", "/-1_0.5_-2_1_0", "/-1_0.5_-2_1_0.25",
                 "/-1_1_-2_2_-0.25", "/-1_1_-2_2_0", "/-1_1_-2_2_0.25")
foldernames <- foldernames[task_id]
designnames <- c("/SRS", "/BLS")

setwd("/nesi/project/uoa03789/PhD/SamplingDesigns")
source("./SurvivalData/generateImputeFuns.R")
n <- 100
pb <- txtProgressBar(min = 0, max = n, initial = 0)
if(!dir.exists('./SurvivalData/SurvivalSample/MICE')){system('mkdir ./SurvivalData/SurvivalSample/MICE')}
if(!dir.exists('./SurvivalData/SurvivalSample/MIXGB')){system('mkdir ./SurvivalData/SurvivalSample/MIXGB')}
if(!dir.exists('./SurvivalData/SurvivalSample/GANs')){system('mkdir ./SurvivalData/SurvivalSample/GANs')}
if(!dir.exists('./SurvivalData/SurvivalSample/cycleGANs')){system('mkdir ./SurvivalData/SurvivalSample/cycleGANs')}
for (j in foldernames){
  cat("Beta:", j, "\n")
  for (i in n){
    cat("Iteration:", i, "\n")
    digit <- str_pad(i, nchar(4444), pad=0)
    for (z in designnames){
      data <- read.csv(paste0("./SurvivalData/SurvivalSample", j, z, z, "_", digit, ".csv"))
      curr_path <- paste0('./SurvivalData/SurvivalSample/MICE', j, z)
      if(!dir.exists(curr_path)){dir.create(curr_path, recursive = T)}
      generateMiceImpute(data, digit, curr_path, type = "survival")

      curr_path <- paste0('./SurvivalData/SurvivalSample/MIXGB', j, z)
      if(!dir.exists(curr_path)){dir.create(curr_path, recursive = T)}
      generateMixgbImpute(data, digit, curr_path, type = "survival")
    }
  }
}