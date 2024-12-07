pacman::p_load("survey", "readxl", "stringr", "dplyr", "purrr", "survival", "tidyr", "ggplot2")
read_excel_allsheets <- function(filename, tibble = FALSE) {
  sheets <- readxl::excel_sheets(filename)
  x <- lapply(sheets, function(X) readxl::read_excel(filename, sheet = X))
  if(!tibble) x <- lapply(x, as.data.frame)
  names(x) <- sheets
  x
}

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


### Copied from Giganti
SurvivalEst<-function(stuff){
  mintime<-min(stuff$time)
  #To calculate survival at a given timepoint, we use the survival estimate from the closest prior time
  #In smaller subsets, there may not be any survival estimates at prior timepoints
  #In those cases, we assign a survival estimate of 1 (use max SE here - questionable appropriateness)
  
  exclude<-sum(times<mintime)
  surv.est<-se.est<-rep(NA,length(times))
  #11MAR2015 - only extract estimates if the KM was actually calculated!
  if (stuff$table["events"]!=0 & mintime<=max(times)){
    for (i in (1+exclude):length(times)){
      surv.est[i]<-stuff$surv[stuff$time==max(stuff$time[stuff$time<=times[i]])]
      se.est[i]<-stuff$std.err[stuff$time==max(stuff$time[stuff$time<=times[i]])]
    }
  }
  surv.est<-ifelse(is.na(surv.est),1,surv.est)
  se.est<-ifelse(is.na(se.est),max(stuff$std.err),se.est)
  
  output<-data.frame(times,surv.est,se.est)
  #If min time > 5, return survival est of 1
  return(output)
}


find_coef_var <- function(imp, sample, type, design){
  m_coefs <- NULL
  m_vars <- NULL
  inclusion <- NULL
  for (m in 1:length(imp)){
    if (type == "diffusion"){
      ith_imp <- imp[[m]]
      sample_diff <- sample
      phase1inds <- sample_diff$R == 0
      if (design == "/SRS"){
        ###--------------Phase-1 Columns [9, 3, 14, 2, 4, 7, 11, 16, 17, 22, 24, 25]--------------
        ###--------------Phase-2 Columns [10, 13, 15, 19, 18, 8, 12, 21, 20, 23, 26, 27]--------------
        cols <- c(10, 13, 15, 19, 18, 8, 12, 21, 20, 23, 26, 27) + 1
      }else if (design == "/BLS"){
        ###--------------Phase-1 Columns [12, 6, 17, 5, 7, 10, 14, 19, 20, 25, 0, 27]--------------
        ###--------------Phase-2 Columns [13, 16, 18, 22, 21, 11, 15, 24, 23, 26, 28, 29]--------------
        cols <- c(13, 16, 18, 22, 21, 11, 15, 24, 23, 26, 28, 29) + 1
      }
      phase1_cols <- c("A.star", "D.star", "lastC.star", 
                       "FirstOImonth.star", "FirstARTmonth.star",
                       "AGE_AT_LAST_VISIT.star", "C.star", 
                       "ARTage.star", "OIage.star", "last.age.star", 
                       "ade.star", "fu.star")
      phase2_cols <- c("A", "D", "lastC", 
                       "FirstOImonth", "FirstARTmonth",
                       "AGE_AT_LAST_VISIT", "C", 
                       "ARTage", "OIage", "last.age", 
                       "ade", "fu")
      for (col in 1:length(cols)){
        #sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * sd(sample_diff[[phase2_cols[col]]], na.rm = T) + mean(sample_diff[[phase2_cols[col]]], na.rm = T)
        sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * (max(sample_diff[[phase2_cols[col]]], na.rm = T) + 1) - 1
      }
      
      replicate <- reCalc(sample_diff)
      
      st1 <- exclude(replicate, FirstARTmonth = "FirstARTmonth", 
                     OIage = "OIage", ARTage = "ARTage", 
                     fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT")
      
      imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = st1, y = FALSE)
      
    }else{
      ith_imp <- imp[[m]]
      replicate <- reCalc(ith_imp)
      
      st1 <- exclude(replicate, FirstARTmonth = "FirstARTmonth", 
                     OIage = "OIage", ARTage = "ARTage", 
                     fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT")
      
      imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = st1, y = FALSE)
    }
    m_coefs <- rbind(m_coefs, imp_mod.1$coef['X'])
    m_vars <- rbind(m_vars, diag(vcov(imp_mod.1))['X'])
    
    inclusion <- rbind(inclusion, nrow(st1))
  }
  
  var <- 1/20 * colSums(m_vars) + (20 + 1) * apply(m_coefs, 2, var) / 20
  return (list(coef = colMeans(m_coefs), var = var,
               inclusion = colMeans(inclusion)))
}


foldernames <- c("/-1_0_-2_0_-0.25", "/-1_0_-2_0_0", "/-1_0_-2_0_0.25",
                 "/-1_0.25_-2_0.5_-0.25", "/-1_0.25_-2_0.5_0", "/-1_0.25_-2_0.5_0.25",
                 "/-1_0.5_-2_1_-0.25", "/-1_0.5_-2_1_0", "/-1_0.5_-2_1_0.25",
                 "/-1_1_-2_2_-0.25", "/-1_1_-2_2_0", "/-1_1_-2_2_0.25")
designnames <- c("/SRS", "/BLS")
n <- 100
result_df <- vector("list", n * length(foldernames) * length(designnames))
m <- 1
pb <- txtProgressBar(min = 0, max = n, initial = 0) 

source("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/generateGigantiData.R")
for (i in n){
  setTxtProgressBar(pb, i)
  digit <- str_pad(i, nchar(4444), pad=0)
  for (j in 1:length(foldernames)){
    for (z in 1:length(designnames)){
      if (!file.exists(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/MECSDI/imputations", 
                              foldernames[j], designnames[z], designnames[z], "_", digit, ".xlsx"))){
        next
      }
      load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/Output", foldernames[j], "/SurvivalData_", digit, ".RData"))
      curr_sample <- read.csv(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalSample", 
                                     foldernames[j], designnames[z], designnames[z], "_", digit, ".csv"))
      
      diff_imp <- read_excel_allsheets(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/MECSDI/imputations", 
                                              foldernames[j], designnames[z], designnames[z], "_", digit, ".xlsx"))
      imp_coefs_vars.diff <- find_coef_var(imp = diff_imp, sample = curr_sample, type = "diffusion", design = designnames[z])
      
      truth <- generateUnDiscretizeData(alldata)
      truth <- exclude(truth)
      true <- coxph(Surv(fu, ade) ~ X, data = truth, y = FALSE)
      
      curr_sample <- exclude(curr_sample[curr_sample$R == 1, ])
      complete <- coxph(Surv(fu, ade) ~ X, data = curr_sample, y = FALSE)
      
      load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalSample/MICE", 
                  foldernames[j], designnames[z], "/MICE_IMPUTE_", digit, ".RData"))
      imp_coefs_vars.mice <- find_coef_var(imp = imputed_data_list, sample = curr_sample, type = "mice", design = designnames[z])
      load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalSample/MIXGB", 
                  foldernames[j], designnames[z], "/MIXGB_IMPUTE_", digit, ".RData"))
      imp_coefs_vars.mixgb <- find_coef_var(imp = imputed_data_list, sample = curr_sample, type = "mixgb", design = designnames[z])
      
      curr_res <- data.frame(TRUE.Est = rep(true$coef['X'], 7),
                             COMPL.Est = rep(complete$coef['X'], 7),
                             MICE.imp.Est = imp_coefs_vars.mice$coef, 
                             MIGXB.imp.Est = imp_coefs_vars.mixgb$coef, 
                             DIFF.imp.Est = imp_coefs_vars.diff$coef,
                               
                             TRUE.Var = rep(diag(vcov(true))['X'], 7),
                             COMPL.Var = rep(diag(vcov(complete))['X'], 7),
                             MICE.imp.Var = imp_coefs_vars.mice$var, 
                             MIGXB.imp.Var = imp_coefs_vars.mixgb$var, 
                             DIFF.imp.Var = imp_coefs_vars.diff$var,
                             
                             MICE.Inc = imp_coefs_vars.mice$inclusion,
                             MIGXB.Inc = imp_coefs_vars.mixgb$inclusion,
                             DIFF.Inc = imp_coefs_vars.diff$inclusion,
                               
                             TYPE = c("RAW", paste0("ST", 1:6)),
                             PARAMS = foldernames[j],
                             DESIGN = designnames[z],
                             DIGIT = digit)
      result_df[[m]] <- curr_res
      m <- m + 1
    }
  }
}

save(result_df, file = "/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalSample/result_imputation.RData")

load("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalSample/result_imputation.RData")
combined_df <- bind_rows(result_df) %>%
  pivot_longer(
    cols = 1:10,
    names_to = c("METHOD", "EST/VAR"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

combined_df.coefs <- combined_df %>%
  filter(`EST/VAR` == "Est", TYPE == "RAW")
true_coefs <- data.frame(DIGIT = combined_df.coefs$DIGIT[combined_df.coefs$METHOD == "TRUE"], 
                         TRUTH = combined_df.coefs$value[combined_df.coefs$METHOD == "TRUE"])

combined_df.coefs <- combined_df.coefs %>%
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs, by = "DIGIT")

rmse <- combined_df.coefs %>% 
  dplyr::group_by(METHOD, DESIGN) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))

#Good Performance experienced a converged training loss at round 0.1, while Bad performance experienced a training loss at 0.14
ggplot(combined_df %>% filter(`EST/VAR` == "Est", TYPE == "RAW")) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "MIGXB.imp", "DIFF.imp")), y = value, colour = DESIGN)) + 
  facet_wrap(~PARAMS, scales = "free") + 
  theme_minimal() + 
  labs(x = "Methods", y = "Estimate", colour = "Sampling Designs") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("TRUE" = "True",  "COMPL" = "Complete-Case",
                              "MICE.imp" = "MICE", "MIGXB.imp" = "MIGXB",
                              "DIFF.imp" = "DIFFUSION")) + 
  scale_color_brewer(palette = "Dark2")



ggplot(rmse) + 
  geom_line(aes(x = as.numeric(factor(DESIGN, levels = c("/SRS", "/RS", "/WRS", "/SFS", 
                                                         "/ODS_extTail", "/SSRS_exactAlloc", 
                                                         "/ODS_exactAlloc", "/SFS_exactAlloc"))), 
                y = rmse, colour = METHOD)) + 
  theme_minimal() +
  scale_x_continuous(breaks = 1:8,
                     labels = c("1" = "/SRS", "2" = "/RS",
                                "3" = "/WRS", "4" = "/SFS",
                                "5" = "/ODS_extTail",
                                "6" = "/SSRS_exactAlloc",
                                "7" = "/ODS_exactAlloc",
                                "8" = "/SFS_exactAlloc")) + 
  labs(x = "Designs", y = "Root Mean Squared Errors", colour = "Methods") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) + 
  scale_color_brewer(palette = "Dark2")
