library(torch)
library(progress)

gain <- function(data, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                 alpha = 10, beta = 1, n = 10000){
  device <- torch_device(device)
  D_loss_mat <- G_loss_mat <- matrix(NA, nrow = n, ncol = 2)
  
  R <- as.matrix(as.numeric(!is.na(rowSums(data))))
  misRow <- R == 0
  R <- torch::torch_tensor(as.matrix(as.numeric(!is.na(rowSums(data)))), 
                           device = device)
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  
  H_dim <- nCol
  
  numCol <- lapply(apply(data, 2, unique), length) > 2
  N <- t(replicate(batch_size, as.numeric(numCol)))
  N <- torch_tensor(N, device = device)
  
  misCol <- as.numeric(!is.na(colSums(data)))
  C <- t(replicate(batch_size, as.numeric(misCol)))
  
  norm_result <- normalize(data, numCol)
  #phase2vals <- norm_result$norm_data[which(misRow == F), which(misCol == 0)]
  #print(phase2vals)
  
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  #norm_data[is.na(norm_data)] <- 0
  
  data_mask <- 1 - is.na(data)
  norm_data <- as.matrix(norm_data)
  
  #norm_data <- torch::torch_tensor(norm_data, device = device)
  data_mask <- torch::torch_tensor(data_mask, device = device)
  
  #norm_data <- torch_data(norm_data)
  data_mask <- torch_data(data_mask)
  
  GAIN_Generator <- torch::nn_module(
    initialize = function(nCol, H_dim){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol + 1, H_dim),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(H_dim, H_dim),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(H_dim, nCol),
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
    initialize = function(nCol, H_dim){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol + 1, H_dim),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(H_dim, H_dim),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(H_dim, 1),
                          name = "Linear3")
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Output")
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  G_layer <- GAIN_Generator(nCol, H_dim)$to(device = device)
  D_layer <- GAIN_Discriminator(nCol, H_dim)$to(device = device)
  
  G_solver <- torch::optim_adam(G_layer$parameters)
  D_solver <- torch::optim_adam(D_layer$parameters)
  
  generator <- function(X, R){
    input <- torch_cat(list(X, R), dim = 2)
    return (G_layer(input))
  }
  discriminator <- function(X, H){
    input <- torch_cat(list(X, H), dim = 2)
    return (D_layer(input))
  }
  
  G_loss <- function(X, M, H, R){
    G_sample <- generator(X, R)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    G_loss1 <- -torch_mean((1 - R) * torch_log(D_prob + 1e-8)) / torch_mean(1 - R)
    mse_loss <- torch_mean((M * X * N - M * G_sample * N) ^ 2) / torch_mean(M * N)
    #mse_loss2 <- torch_mean((M * X * N * (1 - C) - M * G_sample * N * (1 - C)) ^ 2) / torch_mean(M * N * (1 - C))
    
    cross_loss <- -torch_mean((1 - N) * X * M * 
                                torch_log(G_sample + 1e-8) + 
                                (1 - X) * (1 - N) * M * torch_log(1 - (G_sample + 1e-8)))
    #print(G_loss1)
    return (G_loss1 + alpha * mse_loss + beta * cross_loss)
  }
  D_loss <- function(X, M, H, R){
    G_sample <- generator(X, R)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    D_loss1 <- -torch_mean(R * torch_log(D_prob + 1e-8) + (1 - R) * torch_log(1 - (D_prob + 1e-8))) * 2
    #D_loss1 <- -torch_mean(M * torch_log(D_prob + 1e-8) + (1 - M) * torch_log(1 - D_prob + 1e-8))
    return (D_loss1)
  }
  
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = n, width = 60)
  
  for (i in 1:n){
    ind_batch <- new_batch(norm_data, data_mask, R, misRow, batch_size, device)
    X_mb <- ind_batch[[1]]
    M_mb <- ind_batch[[2]]$to(device = device)
    R_mb <- ind_batch[[3]]$to(device = device)
    

    X_mb <- torch_tensor(X_mb, device = device)
    
    Z_mb <- ((-0.01) * torch::torch_rand(c(batch_size, nCol)) + 0.01)$to(device = device)
    H_mb <- 1 * (matrix(runif(batch_size * 1, 0, 1), nrow = batch_size) < hint_rate)
    H_mb <- torch_tensor(H_mb, device = device)
    
  
    #H_mb <- M_mb * H_mb
    H_mb <- R_mb * H_mb
    X_mb <- M_mb * X_mb + (1 - M_mb) * (Z_mb)
    X_mb <- X_mb$to(device = device)
    
    d_loss <- D_loss(X_mb, M_mb, H_mb, R_mb)
    D_loss_mat[i, ] <- c(i, as.numeric(d_loss$detach()$cpu()))
  
    D_solver$zero_grad()
    d_loss$backward()
    D_solver$step()
    
    g_loss <- G_loss(X_mb, M_mb, H_mb, R_mb)
    G_loss_mat[i, ] <- c(i, as.numeric(g_loss$detach()$cpu()))
    
    G_solver$zero_grad()
    g_loss$backward()
    G_solver$step()
    
    pb$tick(tokens = list(what = "GAIN   "))
    Sys.sleep(1 / 10000)
  }
  norm_data <- torch_tensor(norm_data, device = device)
  Z <- ((-0.01) * torch::torch_rand(c(nRow, nCol)) + 0.01)$to(device = device)
  X <- data_mask$torch.data * norm_data + (1 - data_mask$torch.data) * Z
  X <- X$to(device = device)
  
  G_sample <- generator(norm_data, R)
  
  imputed_data <- data_mask$torch.data * norm_data + (1 - data_mask$torch.data) * G_sample
  
  imputed_data <- imputed_data$detach()$cpu()
  
  imputed_data <- renormalize(imputed_data, norm_parameters, numCol)
  
  imputed_data <- data.frame(as.matrix(imputed_data))
  
  names(imputed_data) <- names(data)

  G_sample <- G_sample$detach()$cpu()
  
  G_sample <- renormalize(G_sample, norm_parameters, numCol)
  
  G_sample <- data.frame(as.matrix(G_sample))
  
  names(G_sample) <- names(data)
  
  D_loss_mat <- as.data.frame(D_loss_mat)
  D_loss_mat$Type <- "D"
  names(D_loss_mat) <- c("epochs", "loss", "Type")
  G_loss_mat <- as.data.frame(G_loss_mat)
  G_loss_mat$Type <- "G"
  names(G_loss_mat) <- c("epochs", "loss", "Type")
  bind_loss <- rbind(D_loss_mat, G_loss_mat)
  
  return (list(imputed_data, G_sample, bind_loss))
}