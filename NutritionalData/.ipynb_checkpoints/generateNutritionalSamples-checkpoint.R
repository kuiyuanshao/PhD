pacman::p_load("stringr", "dplyr", "data.table")
source("/nesi/project/uoa03789/PhD/SamplingDesigns/exactAlloc.R")
generateSamples <- function(population, population_name, n_phase2, stratum_variable, outcome_variable, target_variables_1, target_variables_2, model_formula, digit){
  nrow <- dim(population)[1]
  
  mainDir <- getwd()
  dir.create(file.path(mainDir, population_name), showWarnings = FALSE)
  
  dir.create(file.path(mainDir, paste0(population_name, "/SRS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0(population_name, "/RS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0(population_name, "/WRS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0(population_name, "/SFS")), showWarnings = FALSE)
  
  dir.create(file.path(mainDir, paste0(population_name, "/ODS_extTail")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0(population_name, "/SSRS_exactAlloc")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0(population_name, "/ODS_exactAlloc")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0(population_name, "/SFS_exactAlloc")), showWarnings = FALSE)
  
  # SRS
  id_phase2 <- sample(nrow, n_phase2)
  data_srs <- population %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_srs, file = paste0(file.path(mainDir, paste0(population_name, "/SRS")), "/", "SRS_", digit, ".csv"))
  
  # ODS with extreme tails
  order_outcome <- order(population[[outcome_variable]])
  id_phase2 <- c(order_outcome[1:(n_phase2 %/% 2)], order_outcome[(nrow - n_phase2 %/% 2 + 1):nrow])
  data_ods <- population %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_ods, file = paste0(file.path(mainDir, paste0(population_name, "/ODS_extTail")), "/", "ODS_extTail_", digit, ".csv"))
  
  # RS
  modphase1 <- lm(as.formula(paste0(model_formula, " + ", target_variables_1[[1]])), data = population)
  rs <- residuals(modphase1)
  order_residual <- order(rs)
  id_phase2 <- c(order_residual[1:(n_phase2 %/% 2)], order_residual[(nrow - n_phase2 %/% 2 + 1):nrow])
  data_rs <- population %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_rs, file = paste0(file.path(mainDir, paste0(population_name, "/RS")), "/", "RS_", digit, ".csv"))
  
  # WRS
  vart_mat <- NULL
  for (j in 1:length(target_variables_1)){
    vs <- NULL
    for (t in unique(population[[stratum_variable]])){
      v <- sd(residuals(lm(as.formula(paste0(target_variables_2[j], " ~ ", 
                                               sub(".*~", "", model_formula), 
                                               " + ", target_variables_1[j])), 
                           data = population))[population[[stratum_variable]] == t])
      vs <- c(vs, v)
    }
    vart_mat <- rbind(vart_mat, vs)
  }
  wrs <- rs * vart_mat[1, population[[stratum_variable]]]
  order_weighted_residual <- order(wrs)
  id_phase2 <- c(order_weighted_residual[1:(n_phase2 %/% 2)], order_weighted_residual[(nrow - n_phase2 %/% 2 + 1):nrow])
  data_wrs <- population %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_wrs, file = paste0(file.path(mainDir, paste0(population_name, "/WRS")), "/", "WRS_", digit, ".csv"))
  
  # SFS
  score <- residuals(modphase1) * population[[target_variables_1[1]]]
  order_score <- order(score)
  id_phase2 <- c(order_score[1:(n_phase2 %/% 2)], order_score[(nrow - n_phase2 %/% 2 + 1):nrow])
  data_sfs <- population %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_sfs, file = paste0(file.path(mainDir, paste0(population_name, "/SFS")), "/", "SFS_", digit, ".csv"))
  
  # SFS with exact allocation
  quantile_split <- c(0.19, 0.81)
  scores_phase1_cut <- cut(score, breaks = c(-Inf, quantile(score, probs = quantile_split), Inf), labels = paste(1:(length(quantile_split) + 1), sep=','))
  data_sfs_exactAlloc <- population
  data_sfs_exactAlloc$sfs_strata <- scores_phase1_cut
  alloc <- exactAllocation(data = data_sfs_exactAlloc, stratum_variable = "sfs_strata", 
                           target_variable = target_variables_1[1], sample_size = n_phase2, type = 2)
  weights <- (table(data_sfs_exactAlloc[["sfs_strata"]]) / alloc)
  id_phase2 <- lapply(1:length(table(data_sfs_exactAlloc[["sfs_strata"]])), function(j){sample((1:nrow)[data_sfs_exactAlloc[["sfs_strata"]] == j], alloc[j])})
  id_phase2 <- unlist(id_phase2)
  data_sfs_exactAlloc <- data_sfs_exactAlloc %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = case_when(!!!lapply(names(weights), function(value){
                                expr(.data[["sfs_strata"]] == !!value ~ !!weights[[value]])
                                })),
           across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_sfs_exactAlloc, file = paste0(file.path(mainDir, paste0(population_name, "/SFS_exactAlloc")), "/", "SFS_exactAlloc_", digit, ".csv"))
  
  # SSRS with exact allocation
  alloc <- exactAllocation(data = population, stratum_variable = stratum_variable, 
                           target_variable = target_variables_1[1], sample_size = n_phase2, type = 2)
  weights <- table(population[[stratum_variable]]) / alloc
  data_ssrs_exactAlloc <- population
  id_phase2 <- lapply(1:length(table(data_ssrs_exactAlloc[[stratum_variable]])), 
                      function(j){sample((1:nrow)[data_ssrs_exactAlloc[[stratum_variable]] == j], alloc[j])})
  id_phase2 <- unlist(id_phase2)
  
  data_ssrs_exactAlloc <- data_ssrs_exactAlloc %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = case_when(!!!lapply(names(weights), function(value){
                                expr(.data[[stratum_variable]] == !!value ~ !!weights[[value]])
                                })),
                  across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_ssrs_exactAlloc, file = paste0(file.path(mainDir, paste0(population_name, "/SSRS_exactAlloc")), "/", "SSRS_exactAlloc_", digit, ".csv"))
  
  # ODS with exact allocation
  quantile_split <- c(0.19, 0.81)
  outcome <- cut(population[[outcome_variable]], breaks = c(-Inf, quantile(population[[outcome_variable]], probs = quantile_split), Inf), 
                 labels = paste(1:(length(quantile_split) + 1), sep=','))
  data_ods_exactAlloc <- population
  data_ods_exactAlloc$outcome_strata <- as.numeric(outcome)
  alloc <- exactAllocation(data = data_ods_exactAlloc, stratum_variable = "outcome_strata", 
                           target_variable = target_variables_1[1], sample_size = n_phase2, type = 2)
  weights <- table(data_ods_exactAlloc[["outcome_strata"]]) / alloc
  id_phase2 <- lapply(1:length(table(data_ods_exactAlloc[["outcome_strata"]])), function(j){sample((1:nrow)[data_ods_exactAlloc[["outcome_strata"]] == j], alloc[j])})
  id_phase2 <- unlist(id_phase2)
  data_ods_exactAlloc <- data_ods_exactAlloc %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = case_when(!!!lapply(names(weights), function(value){
                                expr(.data[["outcome_strata"]] == !!value ~ !!weights[[value]])
                                })),
           across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_ods_exactAlloc, file = paste0(file.path(mainDir, paste0(population_name, "/ODS_exactAlloc")), "/", "ODS_exactAlloc_", digit, ".csv"))
  
  # RS with exact allocation
  rs_cut <- cut(rs, breaks = c(-Inf, quantile(rs, probs = quantile_split), Inf), 
                labels = paste(1:(length(quantile_split) + 1), sep=','))
  data_rs_exactAlloc <- population
  data_rs_exactAlloc$wrs_strata <- rs_cut
  alloc <- exactAllocation(data = data_rs_exactAlloc, stratum_variable = "wrs_strata", 
                           target_variable = target_variables_1[1], sample_size = n_phase2, type = 2)
  weights <- table(data_sfs_exactAlloc[["rs_strata"]]) / alloc
  id_phase2 <- lapply(1:length(table(data_rs_exactAlloc[["rs_strata"]])), function(j){sample((1:nrow)[data_rs_exactAlloc[["rs_strata"]] == j], alloc[j])})
  id_phase2 <- unlist(id_phase2)
  data_wrs_exactAlloc <- data_sfs_exactAlloc %>%
    mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
           W = case_when(!!!lapply(names(weights), function(value){
             expr(.data[["wrs_strata"]] == !!value ~ !!weights[[value]])
           })),
           across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_wrs_exactAlloc, file = paste0(file.path(mainDir, paste0(population_name, "/WRS_exactAlloc")), "/", "WRS_exactAlloc_", digit, ".csv"))
  
  # WRS with exact allocation
  wrs_cut <- cut(wrs, breaks = c(-Inf, quantile(wrs, probs = quantile_split), Inf), 
                 labels = paste(1:(length(quantile_split) + 1), sep=','))
  data_wrs_exactAlloc <- population
  data_wrs_exactAlloc$wrs_strata <- wrs_cut
  alloc <- exactAllocation(data = data_wrs_exactAlloc, stratum_variable = "wrs_strata", 
                           target_variable = target_variables_1[1], sample_size = n_phase2, type = 2)
  weights <- table(data_sfs_exactAlloc[["wrs_strata"]]) / alloc
  id_phase2 <- lapply(1:length(table(data_wrs_exactAlloc[["wrs_strata"]])), function(j){sample((1:nrow)[data_wrs_exactAlloc[["wrs_strata"]] == j], alloc[j])})
  id_phase2 <- unlist(id_phase2)
  data_wrs_exactAlloc <- data_wrs_exactAlloc %>%
    mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
           W = case_when(!!!lapply(names(weights), function(value){
                         expr(.data[["wrs_strata"]] == !!value ~ !!weights[[value]])
                         })),
           across(all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_wrs_exactAlloc, file = paste0(file.path(mainDir, paste0(population_name, "/WRS_exactAlloc")), "/", "WRS_exactAlloc_", digit, ".csv"))
}

setwd("/nesi/project/uoa03789/PhD/SamplingDesigns")
# The stratum variable is assumed to be a numeric strata, and the minimum is assumed to be 1.

# Nutritional Data
n <- 100
pb <- txtProgressBar(min = 0, max = n, initial = 0) 
for (i in 1:n){
  set.seed(i)
  setTxtProgressBar(pb, i)
  digit <- str_pad(i, nchar(4444), pad=0)
  generateNutritionalData(digit)
  load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/Output/NutritionalData_", digit, ".RData"))
  
  generateSamples(as.data.frame(samp), "NutritionalSample", 1000, "idx", "sbp", 
                  c("c_ln_na_bio1", "c_ln_k_bio1", "c_ln_kcal_bio1", "c_ln_protein_bio1"), 
                  c("c_ln_na_true", "c_ln_k_true", "c_ln_kcal_true", "c_ln_protein_true"), 
                  "sbp ~ c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr", digit)
}
close(pb)



# Simple Linear Data
#source("/nesi/project/uoa03789/PhD/SamplingDesigns/SimpleLinearData/generateLinearData.R")
#n <- 1000
#pb <- txtProgressBar(min = 0, max = n, initial = 0) 
#for (i in 1:n){
#  setTxtProgressBar(pb, i)
#  digit <- str_pad(i, nchar(4444), pad=0)
#  generateLinearData(digit)
#  load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/SimpleLinearData/Output/SimpleLinearData_", digit, ".RData"))
#  generateSamples(as.data.frame(data), "LinearSample", 0.8, "Z", "Y", 
#                  c("X_tilde"), 
#                  c("X"), 
#                  as.formula("Y ~ Z"), digit)
#}
#close(pb)
                                    


