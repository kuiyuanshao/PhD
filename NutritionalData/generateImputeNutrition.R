pacman::p_load("stringr")

args <- commandArgs(trailingOnly = TRUE)
task_id <- as.numeric(args[1])      # SLURM_ARRAY_TASK_ID
num_tasks <- as.numeric(args[2])   # Total number of tasks (10)
total_iterations <- as.numeric(args[3])
iterations_per_task <- ceiling(total_iterations / num_tasks)
start_iteration <- (task_id - 1) * iterations_per_task + 1
end_iteration <- min(task_id * iterations_per_task, total_iterations)


source("/nesi/project/uoa03789/PhD/SamplingDesigns/generateImputeFuns.R")
# Nutritional Data Imputations
foldernames <- c("/SRS", "/RS", "/WRS", "/SFS", "/ODS_extTail", "/SSRS_exactAlloc", 
                 "/ODS_exactAlloc", "/RS_exactAlloc", "/WRS_exactAlloc", "/SFS_exactAlloc")

if(!dir.exists('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/MICE')){system('mkdir /nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/MICE')}
if(!dir.exists('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/MIXGB')){system('mkdir /nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/MIXGB')}
if(!dir.exists('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/GANs')){system('mkdir /nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/GANs')}
if(!dir.exists('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/cycleGANs')){system('mkdir /nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/cycleGANs')}

for (i in start_iteration:end_iteration){
  cat("Iteration:", i, "\n")
  digit <- str_pad(i, nchar(4444), pad=0)
  for (j in foldernames){
    data <- read.csv(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample",  j, j, "_", digit, ".csv"))
    curr_path <- paste0('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MICE', j)
    if(!dir.exists(curr_path)){system(paste0('mkdir ', curr_path))}
    generateMiceImpute(data, digit, curr_path, type = "nutrition")

    curr_path <- paste0('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MIXGB', j)
    if(!dir.exists(curr_path)){system(paste0('mkdir ', curr_path))}
    generateMixgbImpute(data, digit, curr_path, type = "nutrition")
    
    curr_path <- paste0('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/GANs', j)
    if(!dir.exists(curr_path)){system(paste0('mkdir ', curr_path))}
    generateGansImpute(data, digit, curr_path, j, type = "nutrition")
    
    curr_path <- paste0('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/cycleGANs', j)
    if(!dir.exists(curr_path)){system(paste0('mkdir ', curr_path))}
    generateCycleGansImpute(data, digit, curr_path, j, type = "nutrition")
  }
}






