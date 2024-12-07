pacman::p_load('dplyr', 'plyr','stringr', 'mvtnorm','MASS','data.table', 'sampling')
source("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/generateGigantiData.R")

generateGigantiSamples <- function(population, population_name, missing_ratio = 0.75, id_variable = "CFAR_PID", 
                                   target_variables_1 = c("A.star", "D.star", "lastC.star", 
                                                          "FirstOImonth.star", "FirstARTmonth.star",
                                                          "AGE_AT_LAST_VISIT.star", "C.star", 
                                                          "ARTage.star", "OIage.star", "last.age.star", 
                                                          "ade.star", "fu.star"), 
                                   target_variables_2 = c("A", "D", "lastC", 
                                                          "FirstOImonth", "FirstARTmonth",
                                                          "AGE_AT_LAST_VISIT", "C", 
                                                          "ARTage", "OIage", "last.age", 
                                                          "ade", "fu"),
                                   digit){
  mainDir <- "/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData"
  dir.create(file.path(mainDir, paste0("SurvivalSample")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("SurvivalSample/", population_name)), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("SurvivalSample/", population_name, "/SRS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("SurvivalSample/", population_name, "/BLS")), showWarnings = FALSE)
  n_subject <- length(unique(population[[id_variable]]))
  n_phase2 <- as.integer(n_subject * (1 - missing_ratio))
  
  # D is the ADE, A is the time
  population <- generateUnDiscretizeData(population)
  # SRS
  id_phase2 <- sample(1:n_subject, n_phase2, replace = F)
  data_srs <- population
  data_srs <- data_srs %>%
    dplyr::mutate(R = ifelse(1:n_subject %in% id_phase2, 1, 0),
                  W = 1,
                  dplyr::across(dplyr::all_of(target_variables_2), ~ ifelse(R == 0, NA, .)))
  write.csv(data_srs, file = paste0(file.path(mainDir, paste0("SurvivalSample/", population_name, "/SRS")), "/", "SRS_", digit, ".csv"))
  
  
  # BLS
  # Cut the time and covariate X into pieces:
  data_bls <- population
  quantile_split <- c(1/3)
  fu.star_cut <- ifelse(population$fu.star > 0, 1, 0)
  X_cut <- cut(population[["X"]], breaks = c(-Inf, quantile(population[["X"]], probs = quantile_split), Inf), 
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
  
  write.csv(data_bls, file = paste0(file.path(mainDir, paste0("SurvivalSample/", population_name, "/BLS")), "/", "BLS_", digit, ".csv"))
  
  #
  
}



n <- 100
pb <- txtProgressBar(min = 0, max = n, initial = 0)
beta.X <- c(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
beta.Y <- c(1, 0.5, 0.25, 0, 1, 0.5, 0.25, 0, 1, 0.5, 0.25, 0)
gamma.X <- c(-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2)
gamma.Y <- c(2, 1, 0.5, 0, 2, 1, 0.5, 0, 2, 1, 0.5, 0)
covXY <- c(-0.25, -0.25, -0.25, -0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25)

seed <- 1
for (k in 1:n){
  digit <- str_pad(k, nchar(4444), pad=0)
  setTxtProgressBar(pb, k)
  for (i in 1:length(beta.X)){
    set.seed(seed)
    seed <- seed + 1
    #generateGigantiData(beta.X[i], beta.Y[i], gamma.X[i], gamma.Y[i], covXY[i], digit)
    load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/Output/",
                beta.X[i], "_", beta.Y[i], "_",
                gamma.X[i], "_", gamma.Y[i], "_",
                covXY[i], "/SurvivalData_", digit, ".RData"))
    generateGigantiSamples(alldata, population_name = paste0(beta.X[i], "_", beta.Y[i], "_",
                                                             gamma.X[i], "_", gamma.Y[i], "_",
                                                             covXY[i]), digit = digit)
  }
}

close(pb)