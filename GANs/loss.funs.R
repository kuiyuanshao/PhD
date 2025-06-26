cross_entropy_loss <- function(fake, true, encode_result, phase2_vars){
  cat_phase2 <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% phase2_vars)
  }))]
  loss <- list()
  i <- 1
  for (cat in cat_phase2){
    loss[[i]] <- nnf_cross_entropy(fake[, cat], 
                                   torch_argmax(true[, cat], dim = 2), 
                                   reduction = "mean")
    i <- i + 1
  }
  loss_t <- torch_stack(loss, dim = 1)$sum()
  return (loss_t)
}

activate_binary_cols <- function(fake, encode_result, phase2_vars, tau = 0.2, hard = F){
  cat_phase2 <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% phase2_vars)
  }))]
  for (cat in cat_phase2){
    fake[, cat] <- nnf_gumbel_softmax(fake[, cat], tau = tau, hard = hard)
  }
  return (fake)
}
