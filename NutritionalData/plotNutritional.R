pacman::p_load("ggplot2", "tidyr", "dplyr", "RColorBrewer", "ggh4x")

#### MODEL-BASED RESULTS
load("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/result_imputation.RData")
combined_df.1 <- bind_rows(result_df.1) %>% 
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:12,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

combined_df.2 <- bind_rows(result_df.2) %>% 
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:12,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

means.1 <- combined_df.1 %>% 
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ TYPE, data = ., FUN = mean)

means.2 <- combined_df.2 %>% 
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ TYPE, data = ., FUN = mean)

ggplot(combined_df.1) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "MIGXB.imp", "DIFF.imp", "GANS.imp")), 
                   y = value, colour = factor(DESIGN, levels = c("/SRS", "/SSRS_exactAlloc", "/RS", "/RS_exactAlloc", 
                                                                 "/WRS", "/WRS_exactAlloc", "/ODS_extTail", "/ODS_exactAlloc",
                                                                 "/SFS", "/SFS_exactAlloc")))) + 
  geom_hline(data = means.1, aes(yintercept = value), linetype = "dashed", color = "black") + 
  facet_wrap(~TYPE, scales = "free", 
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) + 
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
                              "DIFF.imp" = "DIFFUSION", "GANS.imp" = "cWGAN-GP")) +
  scale_color_brewer(palette = "Paired") +
  facetted_pos_scales(y = list(TYPE == "Est" ~ scale_y_continuous(limits = c(0, 2.5)),
                               TYPE == "Var" ~ scale_y_continuous(limits = c(0, 0.09))))


ggplot(combined_df.2) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "MIGXB.imp", "DIFF.imp", "GANS.imp")), 
                   y = value, colour = factor(DESIGN, levels = c("/SRS", "/SSRS_exactAlloc", "/RS", "/RS_exactAlloc", 
                                                                 "/WRS", "/WRS_exactAlloc", "/ODS_extTail", "/ODS_exactAlloc",
                                                                 "/SFS", "/SFS_exactAlloc")))) + 
  geom_hline(data = means.2, aes(yintercept = value), linetype = "dashed", color = "black") + 
  facet_wrap(~TYPE, scales = "free", 
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) + 
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
                              "DIFF.imp" = "DIFFUSION", "GANS.imp" = "cWGAN-GP")) + 
  scale_color_brewer(palette = "Paired") + 
  facetted_pos_scales(y = list(TYPE == "Est" ~ scale_y_continuous(limits = c(20, 40)),
                               TYPE == "Var" ~ scale_y_continuous(limits = c(0.2, 1))))

# For Logistic Regression
combined_df.1.coefs <- combined_df.1 %>%
  filter(TYPE == "Est")
true_coefs.1 <- data.frame(DIGIT = combined_df.1.coefs$DIGIT[combined_df.1.coefs$METHOD == "TRUE"], 
                           TRUTH = combined_df.1.coefs$value[combined_df.1.coefs$METHOD == "TRUE"])

combined_df.1.coefs <- combined_df.1.coefs %>%
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs.1, by = "DIGIT")

rmse.1 <- combined_df.1.coefs %>% 
  dplyr::group_by(METHOD, DESIGN) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))

# For Linear Regression
combined_df.2.coefs <- combined_df.2 %>%
  filter(TYPE == "Est")
true_coefs.2 <- data.frame(DIGIT = combined_df.2.coefs$DIGIT[combined_df.2.coefs$METHOD == "TRUE"], 
                           TRUTH = combined_df.2.coefs$value[combined_df.2.coefs$METHOD == "TRUE"])

combined_df.2.coefs <- combined_df.2.coefs %>%
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs.2, by = "DIGIT")

rmse.2 <- combined_df.2.coefs %>% 
  dplyr::group_by(METHOD, DESIGN) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))


### Plot for Logistic Regression
ggplot(rmse.1) + 
  geom_line(aes(x = as.numeric(factor(DESIGN, levels = c("/SRS", "/SSRS_exactAlloc", "/RS", "/RS_exactAlloc", 
                                                         "/WRS", "/WRS_exactAlloc", "/ODS_extTail", "/ODS_exactAlloc",
                                                         "/SFS", "/SFS_exactAlloc"))), 
                y = rmse, colour = METHOD)) + 
  theme_minimal() +
  scale_x_continuous(breaks = 1:10,
                     labels = c("1" = "/SRS", "2" = "/SSRS_exactAlloc",
                                "3" = "/RS", "4" = "/RS_exactAlloc",
                                "5" = "/WRS", "6" = "/WRS_exactAlloc",
                                "7" = "/ODS_extTail", "8" = "/ODS_exactAlloc",
                                "9" = "/SFS", "10" = "/SFS_exactAlloc")) + 
  labs(x = "Designs", y = "Root Mean Squared Errors", colour = "Methods") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) + 
  scale_color_brewer(palette = "Dark2")


### Plot for Linear Regression
ggplot(rmse.2) + 
  geom_line(aes(x = as.numeric(factor(DESIGN, levels = c("/SRS", "/SSRS_exactAlloc", "/RS", "/RS_exactAlloc", 
                                                         "/WRS", "/WRS_exactAlloc", "/ODS_extTail", "/ODS_exactAlloc",
                                                         "/SFS", "/SFS_exactAlloc"))), 
                y = rmse, colour = METHOD)) + 
  theme_minimal() +
  scale_x_continuous(breaks = 1:10,
                     labels = c("1" = "/SRS", "2" = "/SSRS_exactAlloc",
                                "3" = "/RS", "4" = "/RS_exactAlloc",
                                "5" = "/WRS", "6" = "/WRS_exactAlloc",
                                "7" = "/ODS_extTail", "8" = "/ODS_exactAlloc",
                                "9" = "/SFS", "10" = "/SFS_exactAlloc")) + 
  labs(x = "Designs", y = "Root Mean Squared Errors", colour = "Methods") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) + 
  scale_color_brewer(palette = "Dark2")




# Bias in Complete-case estimator is too large, filtering out.
ggplot(rmse.2 %>% filter(METHOD != "COMPL")) + 
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


#### DESIGN-BASED RESULTS vs MODEL-BASED RESULTS
load("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/result_design_based.RData")
combined_df_design.1 <- bind_rows(result_df.1) %>% 
  filter(grepl("^c_ln_na_bio1", rownames(.))) %>%
  pivot_longer(
    cols = 1:6,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )
combined_df.1 <- rbind(combined_df.1, combined_df_design.1)

combined_df_design.2 <- bind_rows(result_df.2) %>% 
  filter(grepl("^c_ln_na_bio1", rownames(.))) %>%
  pivot_longer(
    cols = 1:6,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )
combined_df.2 <- rbind(combined_df.2, combined_df_design.2)

# For Logistic Regression
combined_df_design.1.coefs <- combined_df_design.1 %>%
  filter(TYPE == "Est")
true_coefs_design.1 <- data.frame(DIGIT = combined_df_design.1.coefs$DIGIT[combined_df_design.1.coefs$METHOD == "TRUE"], 
                           TRUTH = combined_df_design.1.coefs$value[combined_df_design.1.coefs$METHOD == "TRUE"])

combined_df_design.1.coefs <- combined_df_design.1.coefs %>%
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs_design.1, by = "DIGIT")

rmse_design.1 <- combined_df_design.1.coefs %>% 
  dplyr::group_by(METHOD, DESIGN) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))

rmse.1 <- rbind(rmse.1, rmse_design.1)

# For Linear Regression
combined_df_design.2.coefs <- combined_df_design.2 %>%
  filter(TYPE == "Est")
true_coefs_design.2 <- data.frame(DIGIT = combined_df_design.2.coefs$DIGIT[combined_df_design.2.coefs$METHOD == "TRUE"], 
                           TRUTH = combined_df_design.2.coefs$value[combined_df_design.2.coefs$METHOD == "TRUE"])

combined_df_design.2.coefs <- combined_df_design.2.coefs %>%
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs_design.2, by = "DIGIT")

rmse_design.2 <- combined_df_design.2.coefs %>% 
  dplyr::group_by(METHOD, DESIGN) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))

rmse.2 <- rbind(rmse.2, rmse_design.2)


ggplot(combined_df.1 %>% filter(DESIGN %in% c("/SRS", "/SSRS_exactAlloc", "/ODS_exactAlloc", "/SFS_exactAlloc"))) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "MIGXB.imp", "DIFF.imp", "IPW", "RAKING")), 
                   y = value, colour = DESIGN)) + 
  facet_wrap(~TYPE, scales = "free", 
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) + 
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
                              "DIFF.imp" = "DIFFUSION",
                              "IPW" = "IPW",
                              "RAKING" = "RAKING")) + 
  scale_color_brewer(palette = "Dark2")

ggplot(combined_df.2 %>% filter(DESIGN %in% c("/SRS", "/SSRS_exactAlloc", "/ODS_exactAlloc", "/SFS_exactAlloc"))) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "MIGXB.imp", "DIFF.imp", "IPW", "RAKING")), y = value, colour = DESIGN)) + 
  facet_wrap(~TYPE, scales = "free", 
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) + 
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
                              "DIFF.imp" = "DIFFUSION",
                              "IPW" = "IPW",
                              "RAKING" = "RAKING")) + 
  scale_color_brewer(palette = "Dark2")
