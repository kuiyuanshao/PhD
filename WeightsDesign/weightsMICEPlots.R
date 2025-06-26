pacman::p_load(stringr)
find_coef_var <- function(imp){
  m_coefs.1 <- NULL
  m_coefs.2 <- NULL
  m_vars.1 <- NULL
  m_vars.2 <- NULL
  for (m in 1:length(imp)){
    ith_imp <- imp[[m]]
    imp_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = binomial())
    imp_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = gaussian())
    m_coefs.1 <- rbind(m_coefs.1, coef(imp_mod.1))
    m_coefs.2 <- rbind(m_coefs.2, coef(imp_mod.2))
    m_vars.1 <- rbind(m_vars.1, diag(vcov(imp_mod.1)))
    m_vars.2 <- rbind(m_vars.2, diag(vcov(imp_mod.2)))
  }
  
  var.1 <- 1/length(imp) * colSums(m_vars.1) + (length(imp) + 1) * apply(m_coefs.1, 2, var) / length(imp)
  var.2 <- 1/length(imp) * colSums(m_vars.2) + (length(imp) + 1) * apply(m_coefs.2, 2, var) / length(imp)
  return (list(coef = list(colMeans(m_coefs.1), colMeans(m_coefs.2)), var = list(var.1, var.2)))
}


#foldernames <- c("/norm", "/wnorm", "/norm_wc", "/wnorm_x", "/norm_wx", "/snorm", 
#                 "/pmm", "/wpmm", "/pmm_wc", "/wpmm_x", "/pmm_wx", "/spmm")
foldernames <- c("/pmm", "/norm", "/wnorm", "/cml", "/cml_rejsamp")
n <- 20
result_df.1 <- vector("list", n * length(foldernames))
result_df.2 <- vector("list", n * length(foldernames))
m <- 1
pb <- txtProgressBar(min = 0, max = n, initial = 0) 
for (i in 1:n){
  setTxtProgressBar(pb, i)
  digit <- str_pad(i, nchar(4444), pad=0)
  for (j in 1:length(foldernames)){
    cat(digit, ":", foldernames[j], "\n")
    load(paste0("../NutritionalData/Output/NutritionalData_", digit, ".RData"))
    curr_sample <- read.csv(paste0("./Test/ODS_exactAlloc/ODS_exactAlloc_", digit, ".csv"))
    
    true.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                    female + bkg_o + bkg_pr, family = binomial(), data = pop)
    true.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                    female + bkg_o + bkg_pr, family = gaussian(), data = pop)
    complete.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                        female + bkg_o + bkg_pr, family = binomial(), data = curr_sample)
    complete.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                        female + bkg_o + bkg_pr, family = gaussian(), data = curr_sample)
    
    load(paste0("./Test/Test_rej", 
                foldernames[j], "/MICE_IMPUTE_", digit, ".RData"))
    imp_coefs_vars.mice <- find_coef_var(imp = imputed_data_list)
    
    curr_res.1 <- data.frame(TRUE.Est = coef(true.1),
                             COMPL.Est = coef(complete.1),
                             MICE.imp.Est = imp_coefs_vars.mice$coef[[1]], 
                             
                             TRUE.Var = diag(vcov(true.1)),
                             COMPL.Var = diag(vcov(complete.1)),
                             MICE.imp.Var = imp_coefs_vars.mice$var[[1]], 
                             
                             TYPE = foldernames[j],
                             DIGIT = digit)
    
    curr_res.2 <- data.frame(TRUE.Est = coef(true.2),
                             COMPL.Est = coef(complete.2),
                             MICE.imp.Est = imp_coefs_vars.mice$coef[[2]], 
                             
                             TRUE.Var = diag(vcov(true.2)),
                             COMPL.Var = diag(vcov(complete.2)),
                             MICE.imp.Var = imp_coefs_vars.mice$var[[2]], 
                             
                             TYPE = foldernames[j],
                             DIGIT = digit)
    result_df.1[[m]] <- curr_res.1
    result_df.2[[m]] <- curr_res.2
    m <- m + 1
  }
}
close(pb)

save(result_df.1, result_df.2, file = "./result_miceTestRej_imputation.RData")

pacman::p_load("ggplot2", "tidyr", "dplyr", "RColorBrewer", "ggh4x")
#### MODEL-BASED RESULTS
load("./result_miceTestRej_imputation.RData")
combined_df.1 <- bind_rows(result_df.1) %>% 
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:6,
    names_to = c("METHOD", "ESTIMATE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  ) %>% 
  mutate(METHOD = case_when(
    METHOD == "MICE.imp" & TYPE == "/pmm"  ~ "MICE.IMP.PMM",
    METHOD == "MICE.imp" & TYPE == "/norm"  ~ "MICE.IMP.NORM",
    METHOD == "MICE.imp" & TYPE == "/wnorm" ~ "MICE.IMP.PWLS",
    METHOD == "MICE.imp" & TYPE == "/cml" ~ "MICE.IMP.CML",
    METHOD == "MICE.imp" & TYPE == "/cml_rejsamp" ~ "MICE.IMP.CML_REJ",
    TRUE ~ METHOD
  ))

combined_df.2 <- bind_rows(result_df.2) %>% 
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:6,
    names_to = c("METHOD", "ESTIMATE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  ) %>% 
  mutate(METHOD = case_when(
    METHOD == "MICE.imp" & TYPE == "/pmm"  ~ "MICE.IMP.PMM",
    METHOD == "MICE.imp" & TYPE == "/norm"  ~ "MICE.IMP.NORM",
    METHOD == "MICE.imp" & TYPE == "/wnorm" ~ "MICE.IMP.PWLS",
    METHOD == "MICE.imp" & TYPE == "/cml" ~ "MICE.IMP.CML",
    METHOD == "MICE.imp" & TYPE == "/cml_rejsamp" ~ "MICE.IMP.CML_REJ",
    TRUE ~ METHOD
  ))


means.1 <- combined_df.1 %>% 
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ ESTIMATE, data = ., FUN = mean)

means.2 <- combined_df.2 %>% 
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ ESTIMATE, data = ., FUN = mean)


ggplot(combined_df.1) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ")), 
                   y = value, colour = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ")))) + 
  geom_hline(data = means.1, aes(yintercept = value), linetype = "dashed", color = "black") + 
  facet_wrap(~ESTIMATE, scales = "free", ncol = 1,
             labeller = labeller(ESTIMATE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Estimate", colour = "Methods") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("TRUE" = "True",  "COMPL" = "Complete-Case",
                              "MICE.IMP.PMM" = "PMM", 
                              "MICE.IMP.NORM" = "NORM", 
                              "MICE.IMP.PWLS" = "PWLS",
                              "MICE.IMP.CML" = "CML",
                              "MICE.IMP.CML_REJ" = "CML_REJ")) +
  scale_color_brewer(palette = "Paired") +
  facetted_pos_scales(y = list(ESTIMATE == "Est" ~ scale_y_continuous(limits = c(0, 2.5)),
                               ESTIMATE == "Var" ~ scale_y_continuous(limits = c(0, 0.09))))

ggsave("Rej_Imputation_logistic_boxplot.png", width = 10, height = 10, limitsize = F)

ggplot(combined_df.2) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ")), 
                   y = value, colour = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ")))) + 
  geom_hline(data = means.2, aes(yintercept = value), linetype = "dashed", color = "black") + 
  facet_wrap(~ESTIMATE, scales = "free", ncol = 1,
             labeller = labeller(ESTIMATE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Estimate", colour = "Methods") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("TRUE" = "True",  "COMPL" = "Complete-Case",
                              "MICE.IMP.PMM" = "PMM", 
                              "MICE.IMP.NORM" = "NORM",
                              "MICE.IMP.PWLS" = "PWLS",
                              "MICE.IMP.CML" = "CML",
                              "MICE.IMP.CML_REJ" = "CML_REJ")) +
  scale_color_brewer(palette = "Paired") + 
  facetted_pos_scales(y = list(ESTIMATE == "Est" ~ scale_y_continuous(limits = c(20, 33)),
                               ESTIMATE == "Var" ~ scale_y_continuous(limits = c(0, 2))))

ggsave("Rej_Imputation_gaussian_boxplot.png", width = 10, height = 10, limitsize = F)




# For Logistic Regression
combined_df.1.coefs <- combined_df.1 %>%
  filter(ESTIMATE == "Est")
true_coefs.1 <- data.frame(DIGIT = combined_df.1.coefs$DIGIT[combined_df.1.coefs$METHOD == "TRUE"], 
                           TRUTH = combined_df.1.coefs$value[combined_df.1.coefs$METHOD == "TRUE"])

combined_df.1.coefs <- combined_df.1.coefs %>%
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs.1, by = "DIGIT")

rmse.1 <- combined_df.1.coefs %>% 
  dplyr::group_by(METHOD) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))
rmse.1
# For Linear Regression
combined_df.2.coefs <- combined_df.2 %>%
  filter(ESTIMATE == "Est")
true_coefs.2 <- data.frame(DIGIT = combined_df.2.coefs$DIGIT[combined_df.2.coefs$METHOD == "TRUE"], 
                           TRUTH = combined_df.2.coefs$value[combined_df.2.coefs$METHOD == "TRUE"])

combined_df.2.coefs <- combined_df.2.coefs %>%
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs.2, by = "DIGIT")

rmse.2 <- combined_df.2.coefs %>% 
  dplyr::group_by(METHOD) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))
rmse.2