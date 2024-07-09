library(torch)
library(progress)

gain <- function(data, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                 alpha = 10, beta = 1, n = 10000){
  device <- torch_device(device)
  D_loss_mat <- G_loss_mat <- matrix(NA, nrow = n, ncol = 2)
  
  R <- as.matrix(as.numeric(!is.na(rowSums(data))))
  misRow <- R == 0
  
  phase1 <- data[misRow, ]
  phase2 <- data[!misRow, ]
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  
  numCol <- lapply(apply(data, 2, unique), length) > 2
  N <- t(replicate(batch_size, as.numeric(numCol)))
  N <- torch_tensor(N, device = device)
  
  misCol <- is.na(colSums(data))
  n_mis <- sum(misCol)
  
  norm_phase1 <- normalize(phase1, numCol)
  norm_phase2 <- normalize(phase2, numCol)
  
  norm_data_phase1 <- norm_phase1$norm_data
  norm_data_phase2 <- norm_phase2$norm_data
  
  norm_parameters_phase1 <- norm_phase1$norm_parameters
  norm_parameters_phase2 <- norm_phase2$norm_parameters
  
  norm_data_phase1 <- as.matrix(norm_data_phase1)
  norm_data_phase2 <- as.matrix(norm_data_phase2)
  
  GAIN_Generator <- torch::nn_module(
    initialize = function(nCol, n_mis){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(nCol, n_mis),
                          name = "Linear3")
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Output")
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  GAIN_Discriminator <- torch::nn_module(
    initialize = function(nCol){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(nCol, 1),
                          name = "Linear3")
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Output")
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  G_layer <- GAIN_Generator(nCol, n_mis)$to(device = device)
  D_layer <- GAIN_Discriminator(nCol)$to(device = device)
  
  G_solver <- torch::optim_adam(G_layer$parameters)
  D_solver <- torch::optim_adam(D_layer$parameters)
  
  generator <- function(X, C){
    input <- torch_cat(list(X, C), dim = 2)
    return (G_layer(input))
  }
  discriminator <- function(X, C){
    input <- torch_cat(list(X, C), dim = 2)
    return (D_layer(input))
  }
  
  G_loss <- function(Z, C, X){
    G_sample <- generator(Z, C)
    D_prob <- discriminator(G_sample, C)
    
    G_loss1 <- -torch_mean(torch_log(D_prob + 1e-8))
    mse_loss <- torch_mean((X - G_sample) ^ 2)
    
    return (G_loss1 + alpha * mse_loss)
  }
  D_loss <- function(Z, C, X){
    G_sample <- generator(Z, C)
    D_prob1 <- discriminator(G_sample, C)
    D_prob2 <- discriminator(X, C)
    D_loss1 <- -torch_mean(torch_log(D_prob2 + 1e-8) + torch_log(1 - D_prob1 + 1e-8))
    return (D_loss1)
  }
  
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = n, width = 60)
  
  for (i in 1:n){
    idx_phase1 <- sample(1:nrow(norm_data_phase1), batch_size)
    phase1_curr_batch <- norm_data_phase1[idx_phase1, ]
    
    idx_phase2 <- sample(1:nrow(norm_data_phase2), batch_size)
    phase2_curr_batch <- norm_data_phase2[idx_phase2, ]
    
    Z_mb <- ((-0.01) * torch::torch_rand(c(batch_size, n_mis)) + 0.01)$to(device = device)
    
    C_mb <- torch_tensor(phase2_curr_batch[, !misCol], device = device)
    X_mb <- torch_tensor(as.matrix(phase2_curr_batch[, misCol]), device = device)
    
    d_loss <- D_loss(Z_mb, C_mb, X_mb)
    D_loss_mat[i, ] <- c(i, as.numeric(d_loss$detach()$cpu()))
  
    D_solver$zero_grad()
    d_loss$backward()
    D_solver$step()
    
    g_loss <- G_loss(Z_mb, C_mb, X_mb)
    G_loss_mat[i, ] <- c(i, as.numeric(g_loss$detach()$cpu()))
    
    G_solver$zero_grad()
    g_loss$backward()
    G_solver$step()
    
    pb$tick(tokens = list(what = "GAIN   "))
    Sys.sleep(1 / 10000)
  }
  #norm_data[misRow, which(misCol == 0)] <- sample(phase2vals, length(which(misRow == T)), replace = T)
  #norm_data[misRow, which(misCol == 0)] <- norm_data[misRow, 1]
  #norm_data <- torch_tensor(norm_data, device = device)
  Z <- ((-0.01) * torch::torch_rand(c(nrow(norm_data_phase1), n_mis)) + 0.01)$to(device = device)
  #X <- data_mask$torch.data * norm_data + (1 - data_mask$torch.data) * Z
  #X <- X$to(device = device)
  
  G_sample <- generator(Z, norm_data_phase1[, !misCol])
  G_sample <- G_sample$detach()$cpu()
  
  G_sample_parameters <- list(min_val = norm_parameters_phase2$min_val[misCol], 
                              max_val = norm_parameters_phase2$max_val[misCol])
  G_sample <- renormalize(as.matrix(G_sample), G_sample_parameters, rep(1, nCol))
  
  imputed_data <- data
  imputed_data[misRow, misCol] <- G_sample
  
  names(imputed_data) <- names(data)
  
  D_loss_mat <- as.data.frame(D_loss_mat)
  D_loss_mat$Type <- "D"
  names(D_loss_mat) <- c("epochs", "loss", "Type")
  G_loss_mat <- as.data.frame(G_loss_mat)
  G_loss_mat$Type <- "G"
  names(G_loss_mat) <- c("epochs", "loss", "Type")
  bind_loss <- rbind(D_loss_mat, G_loss_mat)
  
  return (list(imputed_data, bind_loss))
}