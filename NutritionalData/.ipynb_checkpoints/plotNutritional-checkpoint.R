pacman::p_load("ggplot2", "tidyr")
load("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/result.RData")
### Some Plots

combined_df.1 <- bind_rows(result_df.1) %>% 
  filter(grepl("^c_ln_na_bio1", rownames(.))) %>%
  pivot_longer(
    cols = 1:10,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

combined_df.2 <- bind_rows(result_df.2) %>% 
  filter(grepl("^c_ln_na_bio1", rownames(.))) %>%
  pivot_longer(
    cols = 1:10,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

ggplot(combined_df.1) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "MIGXB.imp", "DIFF.imp")), y = value, colour = DESIGN)) + 
  facet_wrap(~TYPE, scales = "free", 
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal() + 
  labs(title = "Comparison of Methods by Sampling Designs in Coefficients and Variances",
       x = "Methods", y = "Estimate", colour = "Sampling Designs") + 
  theme(plot.title = element_text(family = "Georgia", face = "bold", size = 16, hjust = 0.5),
        axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("TRUE" = "True",  "COMPL" = "Complete-Case",
                              "MICE.imp" = "MICE", "MIGXB.imp" = "MIGXB",
                              "DIFF.imp" = "DIFFUSION")
  )

ggplot(combined_df.2) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "MIGXB.imp", "DIFF.imp")), y = value, colour = DESIGN)) + 
  facet_wrap(~TYPE, scales = "free", 
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal() + 
  labs(title = "Comparison of Methods by Sampling Designs in Coefficients and Variances",
       x = "Methods", y = "Estimate", colour = "Sampling Designs") + 
  theme(plot.title = element_text(family = "Georgia", face = "bold", size = 16, hjust = 0.5),
        axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("TRUE" = "True",  "COMPL" = "Complete-Case",
                              "MICE.imp" = "MICE", "MIGXB.imp" = "MIGXB",
                              "DIFF.imp" = "DIFFUSION")
  )

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
  labs(title = "Comparison of Methods by Sampling Designs in Coefficients MSE",
       x = "Designs", y = "Root Mean Squared Errors", colour = "Methods") + 
  theme(plot.title = element_text(family = "Georgia", face = "bold", size = 16, hjust = 0.5),
        axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia"))


### Plot for Linear Regression
ggplot(rmse.2) + 
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
  labs(title = "Comparison of Methods by Sampling Designs in Coefficients MSE",
       x = "Designs", y = "Root Mean Squared Errors", colour = "Methods") + 
  theme(plot.title = element_text(family = "Georgia", face = "bold", size = 16, hjust = 0.5),
        axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia"))




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
  labs(title = "Comparison of Methods by Sampling Designs in Coefficients MSE",
       x = "Designs", y = "Root Mean Squared Errors", colour = "Methods") + 
  theme(plot.title = element_text(family = "Georgia", face = "bold", size = 16, hjust = 0.5),
        axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia"))





