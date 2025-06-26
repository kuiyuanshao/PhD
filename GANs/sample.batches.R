samplebatches <- function(data_original, data_training, data_mask, phase1_t, phase2_t, phase1_rows, phase2_rows, 
                          phase1_vars, phase2_vars, num_vars, cat_vars, weight_var, batch_size, at_least_p = 0.2){
  #Provide a case-control based depending on phase1 and phase2 logical variables
  phase1_bins <- phase1_vars[!(phase1_vars %in% num_vars)]
  phase2_bins <- phase2_vars[!(phase2_vars %in% num_vars)]
  #sample a binary variate to case-control on in current sample
  curr_col_1 <- sample(phase1_bins, 1)
  curr_col_2 <- sample(phase2_bins, 1)
  
  referring_1 <- data_training[phase1_rows, curr_col_1]
  referring_2 <- data_training[phase2_rows, curr_col_2]
  unicats <- unique(referring_1)
  n_unicats <- length(unicats)
  
  n1 <- batch_size - as.integer(at_least_p * batch_size) #Number of sample to sample at phase1 rows
  n2 <- as.integer(at_least_p * batch_size) #Number of samples to sample at phase2 rows
  n1 <- c(floor(n1 / 2), ceiling(n1 / 2))
  n2 <- c(floor(n2 / 2), ceiling(n2 / 2))
  sampled_ind <- c()
  
  for (cat in 1:n_unicats){
    phase1_sub <- phase1_rows[which(referring_1 == unicats[cat])]
    phase2_sub <- phase2_rows[which(referring_2 == unicats[cat])]
    sampphase1 <- sample(phase1_sub, size = n1[cat],
                         prob = (1 / data_original[phase1_sub, weight_var]) /
                             sum((1 / data_original[phase1_sub, weight_var])))
    sampphase2 <- sample(phase2_sub, size = n2[cat],
                         prob = (1 / data_original[phase2_sub, weight_var]) /
                             sum((1 / data_original[phase2_sub, weight_var])))
    sampled_ind <- c(sampled_ind, sampphase1, sampphase2)
  }
  batches <- list(X = phase2_t[sampled_ind, ],
                  C = phase1_t[sampled_ind, ],
                  M = data_mask[sampled_ind, ])
  return (batches)
}