

pacman::p_load("survey", "readxl", "stringr", "dplyr", "purrr")

read_excel_allsheets <- function(filename, tibble = FALSE) {
  sheets <- readxl::excel_sheets(filename)
  x <- lapply(sheets, function(X) readxl::read_excel(filename, sheet = X))
  if(!tibble) x <- lapply(x, as.data.frame)
  names(x) <- sheets
  x
}

find_coef_var <- function(imp, sample = 0, type = 0, design = 0){
  m_coefs.1 <- NULL
  m_coefs.2 <- NULL
  m_vars.1 <- NULL
  m_vars.2 <- NULL
  for (m in 1:length(imp)){
    if (type == "diffusion"){
      ith_imp <- imp[[m]]
      sample_diff <- sample
      phase1inds <- sample_diff$R == 0
      ##--------------Phase-1 Columns [14, 15, 16, 17]--------------
      ##--------------Phase-2 Columns [10, 11, 12, 13]--------------
      cols <- 5:8 + 1
      phase1_cols <- c("c_ln_na_bio1", "c_ln_k_bio1", 
                       "c_ln_kcal_bio1", "c_ln_protein_bio1")
      phase2_cols <- c("c_ln_na_true", "c_ln_k_true", 
                       "c_ln_kcal_true", "c_ln_protein_true")
      
      for (col in 1:length(cols)){
        #sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * (max(sample_diff[[phase2_cols[col]]], na.rm = T) + 1) - 1
        #sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * 
        #(max(sample_diff[[phase2_cols[col]]], na.rm = T) - min(sample_diff[[phase2_cols[col]]], na.rm = T) + 1) + 
        #(min(sample_diff[[phase2_cols[col]]], na.rm = T) - 1)
        sample_diff[[phase2_cols[col]]][phase1inds] <- (ith_imp[phase1inds, cols[col]]) * 
          sd(sample_diff[[phase2_cols[col]]], na.rm = T) + 
          mean(sample_diff[[phase2_cols[col]]], na.rm = T)
      }
      
      imp_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, sample_diff, family = binomial())
      imp_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, sample_diff, family = gaussian())
      
    }else{
      ith_imp <- imp[[m]]
      
      imp_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = binomial())
      imp_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = gaussian())
    }
    m_coefs.1 <- rbind(m_coefs.1, coef(imp_mod.1))
    m_coefs.2 <- rbind(m_coefs.2, coef(imp_mod.2))
    m_vars.1 <- rbind(m_vars.1, diag(vcov(imp_mod.1)))
    m_vars.2 <- rbind(m_vars.2, diag(vcov(imp_mod.2)))
  }
  
  var.1 <- 1/20 * colSums(m_vars.1) + (20 + 1) * apply(m_coefs.1, 2, var) / 20
  var.2 <- 1/20 * colSums(m_vars.2) + (20 + 1) * apply(m_coefs.2, 2, var) / 20
  return (list(coef = list(colMeans(m_coefs.1), colMeans(m_coefs.2)), var = list(var.1, var.2)))
}


diff_data <- read_excel_allsheets("/nesi/project/uoa03789/PhD/SamplingDesigns/MECSDI/tokenizer/imputations/SFS/SFS_0001.xlsx")
curr_sample <- read.csv("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/SFS/SFS_0001.csv")
imp_coefs_vars.diff <- find_coef_var(imp = diff_data, sample = curr_sample, type = "diffusion", design = "/SRS")



library(ggplot2)
library(cowplot)
data <- read.csv("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalSample/SRS/SRS_0001.csv")
load("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/Output/NutritionalData_0001.RData")


load("./min-max-[-1, 1]_tanh.RData")

loss <- gain_imp$loss

loss_p <- ggplot() + 
  geom_line(aes(x = 1:2000, y = loss$`G Loss`), colour = "red") + 
  geom_line(aes(x = 1:2000, y = loss$`D Loss`), colour = "blue") + 
  theme_minimal() + 
  xlab("Epoch") + ylab("Loss") 

mod <- lm(sbp ~ c_ln_na_true, data = gain_imp$sample[[1]])

combined_plot <- ggdraw() + draw_plot(loss_p, 0, 0, 1, 1)
for (i in c(2, 5, 10, 15, 20)){
  dist_plot <- ggplot(gain_imp$epoch_result[[i]][[1]]) + 
    geom_point(aes(x = c_ln_na_true, y = sbp), alpha = 0.2) + 
    theme_void() +  # Start with a blank theme
    theme(
      axis.title = element_blank(), 
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_blank(), 
      plot.title = element_blank(), 
      plot.subtitle = element_blank() 
    ) + 
    geom_abline(intercept = 120.1, slope = 31, lty = 2, colour = "purple") + 
    geom_smooth(aes(x = c_ln_na_true, y = sbp), method = "lm", se = FALSE, colour = "orange", lty = 2)
  
  
  combined_plot <- combined_plot + 
    draw_plot(dist_plot, x = i / 20 + 0.01, 
              y = loss$`G Loss`[i * 100] / 35, 
              width = 0.3, height = 0.3)
  
  #loss_p <- loss_p + 
  #  annotate("segment", x = i * 100, xend = i * 100 + 200, 
  #           y = loss$`G Loss`[i * 100], yend = loss$`G Loss`[i * 100]) 
  
}
combined_plot





load("./min-max-[0, 1]_linear.RData")

loss <- gain_imp$loss

loss_p <- ggplot() + 
  geom_line(aes(x = 1:3000, y = loss$`G Loss`), colour = "red") + 
  geom_line(aes(x = 1:3000, y = loss$`D Loss`), colour = "blue") + 
  theme_minimal() + 
  xlab("Epoch") + ylab("Loss") 

mod <- lm(sbp ~ c_ln_na_true, data = gain_imp$sample[[1]])

combined_plot <- ggdraw() + draw_plot(loss_p, 0, 0, 1, 1)
for (i in c(2, 5, 10, 15, 20, 25, 30)){
  dist_plot <- ggplot(gain_imp$epoch_result[[i]][[1]]) + 
    geom_point(aes(x = c_ln_na_true, y = sbp), alpha = 0.2) + 
    theme_void() +  # Start with a blank theme
    theme(
      axis.title = element_blank(), 
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_blank(), 
      plot.title = element_blank(), 
      plot.subtitle = element_blank() 
    ) + 
    geom_abline(intercept = 120.1, slope = 31, lty = 2, colour = "purple") + 
    geom_smooth(aes(x = c_ln_na_true, y = sbp), method = "lm", se = FALSE, colour = "orange", lty = 2)
  
  
  combined_plot <- combined_plot + 
    draw_plot(dist_plot, x = i / 30 + 0.01, 
              y = loss$`G Loss`[i * 100] / 5, 
              width = 0.3, height = 0.3)
  
  #loss_p <- loss_p + 
  #  annotate("segment", x = i * 100, xend = i * 100 + 200, 
  #           y = loss$`G Loss`[i * 100], yend = loss$`G Loss`[i * 100]) 
  
}
combined_plot
