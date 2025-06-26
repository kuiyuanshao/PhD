pacman::p_load("ggplot2", "tidyr", "dplyr", "RColorBrewer", "ggh4x")
#### MODEL-BASED RESULTS
load("../SurvivalData/SurvivalSample/result_imputation.RData")
combined_df <- bind_rows(result_df) %>% 
  pivot_longer(
    cols = 1:8,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

means <- combined_df %>%
  dplyr::filter(METHOD == "TRUE") %>%
  dplyr::group_by(PARAMS, TYPE) %>%
  dplyr::summarise(mean_value = mean(value, na.rm = TRUE), .groups = 'drop')

ggplot(combined_df) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "GANs.imp")), 
                   y = value,
                   colour = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "GANs.imp")))) + 
  geom_hline(data = means, 
             aes(yintercept = mean_value), linetype = "dashed", color = "black",
             inherit.aes = FALSE) + 
  facet_grid(cols = vars(PARAMS), rows = vars(TYPE), scales = "free",
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
                              "MICE.imp" = "MICE", "GANs.imp" = "cWGAN-GP")) +
  facetted_pos_scales(y = list(TYPE == "Est" ~ scale_y_continuous(limits = c(-3, -1)),
                               TYPE == "Var" ~ scale_y_continuous(limits = c(0, 0.05)))) + 
  scale_color_brewer(palette = "Dark2")

ggsave("../SurvivalData/Imputation_boxplot.png", width = 30, height = 10, limitsize = F)

combined_df.coefs <- combined_df %>%
  filter(TYPE == "Est")
true_coefs <- data.frame(DIGIT = combined_df.coefs$DIGIT[combined_df.coefs$METHOD == "TRUE"], 
                         PARAMS = combined_df.coefs$PARAMS[combined_df.coefs$METHOD == "TRUE"],
                         TRUTH = combined_df.coefs$value[combined_df.coefs$METHOD == "TRUE"])
combined_df.coefs <- combined_df.coefs %>% 
  filter(METHOD != "TRUE") %>%
  merge(., true_coefs, by = c("DIGIT", "PARAMS"))
rmse <- combined_df.coefs %>%
  dplyr::group_by(METHOD, PARAMS) %>% 
  dplyr::summarize(rmse = sqrt(mean((value - TRUTH)^2)))

for (i in unique(combined_df$PARAMS)){
  print(rmse %>% filter(PARAMS == i))  
}

load("../SurvivalData/SurvivalSample/GANs/-1_0_-2_0_-0.25/SRS/GANs_IMPUTE_0001.RData")
load("../SurvivalData/SurvivalSample/MICE/-1_0_-2_0_-0.25/SRS/MICE_IMPUTE_0001.RData")
load("../SurvivalData/Output/-1_0_-2_0_-0.25/SurvivalData_0001.RData")
curr_sample <- read.csv("../SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/SRS/SRS_0001.csv")
truth <- generateUnDiscretizeData(alldata)
ggplot() + 
  geom_density(aes(x = fu), data = truth) +
  geom_density(aes(x = fu), data = curr_sample, colour = "grey") +
  geom_density(aes(x = fu), data = imputed_data_list[[1]], colour = "blue") +
  geom_density(aes(x = fu), data = gain_imp$imputation[[1]], colour = "red")


