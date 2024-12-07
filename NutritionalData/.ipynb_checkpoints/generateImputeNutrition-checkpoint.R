pacman::p_load("stringr")

foldernames <- c("/SRS", "/RS", "/WRS", "/SFS", "/ODS_extTail", "/SSRS_exactAlloc", "/ODS_exactAlloc", "/SFS_exactAlloc")

# Nutritional Data Imputations
source("/nesi/project/uoa03789/PhD/SamplingDesigns/generateImputeFuns.R")
n <- 100
pb <- txtProgressBar(min = 0, max = n, initial = 0) 
if(!dir.exists('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MICE')){system('mkdir /nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MICE')}
if(!dir.exists('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MIXGB')){system('mkdir /nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MIXGB')}
for (i in 1:n){
  setTxtProgressBar(pb, i)
  digit <- str_pad(i, nchar(4444), pad=0)
  for (j in foldernames){
    data <- read.csv(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample",  j, j,"_", digit, ".csv")) 
    curr_path <- paste0('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MICE', j)
    if(!dir.exists(curr_path)){system(paste0('mkdir ', curr_path))}
    generateMiceImpute(data, digit, curr_path)
    
    curr_path <- paste0('/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/MIXGB', j)
    if(!dir.exists(curr_path)){system(paste0('mkdir ', curr_path))}
    generateMixgbImpute(data, digit, curr_path)
  }
}
close(pb)