library(torch)
library(progress)

generateImpute <- function(generator, m = 5, dat_norm, datainfo, batch_size, device, mse_cols, cross_entropy_cols){
  imputed_data_list <- list()
  sample_data_list <- list()
  impute_batches <- createimputebatches(dat_norm, datainfo, batch_size, device = device)
  for (z in 1:m){
    output_tensor <- NULL
    for (i in 1:ceiling(nrow(dat_norm$data) / batch_size)){
      batch <- impute_batches$batches[[i]]
      X <- batch$X
      A <- batch$A
      C <- batch$C
      M <- batch$M
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 128))$to(device = device)
      fakecat <- torch_cat(list(fakez, A, C), dim = 2)
      G_sample <- generator(fakecat)
      if (is.null(output_tensor)){
        if (length(cross_entropy_cols) > 0){
          binary_act <- nnf_sigmoid(G_sample[, cross_entropy_cols, drop = F])
          G_sample <- torch_cat(list(G_sample[, mse_cols, drop = F], binary_act, A, C), dim = 2)
        }else{
          G_sample <- torch_cat(list(G_sample, A, C), dim = 2)
        }
        output_tensor <- G_sample 
      }else{
        if (length(cross_entropy_cols) > 0){
          binary_act <- nnf_sigmoid(G_sample[, cross_entropy_cols, drop = F])
          G_sample <- torch_cat(list(G_sample[, mse_cols, drop = F], binary_act, A, C), dim = 2)
        }else{
          G_sample <- torch_cat(list(G_sample, A, C), dim = 2)
        }
        output_tensor <- torch_cat(list(output_tensor, G_sample), dim = 1)
      }
    }
    output_tensor <- torch_cat(list(output_tensor, impute_batches$weights), dim = 2)
    imputation <- impute_batches$data_mask * torch_cat(list(impute_batches$phase2_variables, 
                                                            impute_batches$phase1_variables,
                                                            impute_batches$cond_variables, 
                                                            impute_batches$weights), dim = 2) + 
      (1 - impute_batches$data_mask) * output_tensor
    imputation <- as.matrix(imputation$detach()$cpu())
    sample <- as.matrix(output_tensor$detach()$cpu())
    
    imputation <- denormalize(imputation, datainfo, dat_norm, method = "min-max")
    sample <- denormalize(sample, datainfo, dat_norm, method = "min-max")
    
    imputed_data_list[[z]] <- imputation
    sample_data_list[[z]] <- sample
  }
  
  return (list(imputation = imputed_data_list, sample = sample_data_list))
}

samplebatches <- function(info_list, batch_size, at_least_p = 0.2, device = "cpu"){
  list2env(info_list, envir = environment())
  ind_phase1 <- datainfo$phase1_rows
  ind_phase2 <- datainfo$phase2_rows
  
  sampphase2 <- sample(ind_phase2, size = as.integer(at_least_p * batch_size), 
                       prob = dat_norm$data_ori[ind_phase2, datainfo$weight_col] / sum(dat_norm$data_ori[ind_phase2, datainfo$weight_col]))
  sampphase1 <- sample(ind_phase1, size = batch_size - as.integer(at_least_p * batch_size), 
                       prob = dat_norm$data_ori[ind_phase1, datainfo$weight_col] / sum(dat_norm$data_ori[ind_phase1, datainfo$weight_col]))
  
  sampled <- c(sampphase1, sampphase2)
  sampled <- sample(sampled)
  batches <- list(X = phase2_variables[sampled, ],
                  A = phase1_variables[sampled, ],
                  C = cond_variables[sampled, ],
                  M = data_mask[sampled, ],
                  W = weights[sampled, ])
  return (batches)
}

createimputebatches <- function(dat_norm, datainfo, batch_size, device = "cpu"){
  data_mask <- torch_tensor(1 - is.na(dat_norm$data), device = device)
  
  
  cond_variables <- torch_tensor(as.matrix(dat_norm$data[, -c(datainfo$phase1_cols, datainfo$phase2_cols, datainfo$weight_col)]), 
                                 device = device)
  weights <- torch_tensor(as.matrix(dat_norm$data[, datainfo$weight_col]), device = device)
  
  phase1_variables <- torch_tensor(as.matrix(dat_norm$data[, datainfo$phase1_cols]), 
                                   device = device)
  phase2_variables <- dat_norm$data[, datainfo$phase2_cols]
  phase2_variables[is.na(phase2_variables)] <- 0
  phase2_variables <- torch_tensor(as.matrix(phase2_variables), device = device)
  
  n <- ceiling(nrow(dat_norm$data) / batch_size)
  idx <- 1
  batches <- list()
  for (i in 1:n){
    if (i == n){
      batch_size <- nrow(dat_norm$data) - batch_size * (n - 1)
    }
    batches[[i]] <- list(X = phase2_variables[idx:(idx + batch_size - 1), ],
                         A = phase1_variables[idx:(idx + batch_size - 1), ],
                         C = cond_variables[idx:(idx + batch_size - 1), ],
                         M = data_mask[idx:(idx + batch_size - 1), ])
    idx <- idx + batch_size
  }
  
  return (list(batches = batches, 
               data_mask = data_mask, 
               cond_variables = cond_variables, 
               phase1_variables = phase1_variables, 
               phase2_variables = phase2_variables,
               weights = weights))
}

normalize <- function(data, datainfo, method){
  if (method == "min-max"){
    maxs <- apply(data, 2, max, na.rm = T)
    mins <- apply(data, 2, min, na.rm = T)
    dat_norm <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% datainfo$numeric_cols){
        #2 * (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6) - 1
        (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6)
        #0.5 * (1 + (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6))
      }else{
        data[, i]
      }
    }))
    
    return (list(data = dat_norm,
                 maxs = maxs,
                 mins = mins,
                 data_ori = data))
  }
}

denormalize <- function(data, datainfo, params, method){
  if (method == "min-max"){
    maxs <- params$maxs
    mins <- params$mins
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% datainfo$numeric_cols){
        #((data[, i] + 1) / 2) * (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6)
        data[, i]* (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6)
        #(data[, i] * 2 - 1) * (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6)
      }else{
        data[, i]
      }
    }))
    data <- as.data.frame(data)
    names(data) <- datainfo$names
  }
  return (data)
}

getdatainfo <- function(data, data_info){
  list2env(data_info, envir = environment())
  phase1_rows <- which(is.na(rowSums(data)))
  phase2_rows <- which(!is.na(rowSums(data)))
  
  phase1_cols <- which(names(data) %in% phase1_cols)
  phase2_cols <- which(names(data) %in% phase2_cols)
  numeric_cols <- which(!(names(data) %in% categorical_cols))
  weight_col <- which(names(data) == weight_col)
  
  return (list(phase1_rows = phase1_rows, phase2_rows = phase2_rows,
               phase1_cols = phase1_cols, phase2_cols = phase2_cols,
               numeric_cols = numeric_cols, weight_col = weight_col,
               names = names(data)))
}

gradient_penalty <- function(D, real_samples, fake_samples){
  alp <- torch_rand(c(real_samples$size(1), 1, 1))
  alp <- alp$repeat_interleave(real_samples$size(2), dim = 3)
  alp <- alp$reshape(c(-1, real_samples$size(2)))
  interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(T)
  #interpolates <- torch_cat(list(interpolates, C), dim = 2)
  d_interpolates <- D(interpolates)
  fake <- torch_ones(d_interpolates$size())
  fake$requires_grad <- F
  gradients <- torch::autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake,
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]
  gradients <- torch_reshape(gradients, c(gradients$size(1), -1))
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  
  return (gradient_penalty)
}



cyclewgangp <- function(data, m = 5,
                        params = list(batch_size = 128, lambda = 10, alpha = 100, n = 75, g_layers = 4), 
                        sampling_info = list(phase1_cols = "X_tilde", phase2_cols = "X", weight_col = "W", 
                                         categorical_cols = c("R", "Z"), outcome_cols = c("Y")), 
                        device = "cpu"){
  list2env(params, envir = environment())
  list2env(sampling_info, envir = environment())
  
  device <- torch_device(device)
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  new_data <- data[, c(phase2_cols, phase1_cols, setdiff(names(data), c(phase2_cols, phase1_cols, weight_col)), weight_col)]
  
  datainfo <- getdatainfo(new_data, sampling_info)
  dat_norm <- normalize(new_data, datainfo, method = "min-max")
  
  data_mask <- torch_tensor(1 - is.na(dat_norm$data), device = device)
  
  cond_variables <- torch_tensor(as.matrix(dat_norm$data[, -c(datainfo$phase1_cols, datainfo$phase2_cols, datainfo$weight_col)]), device = device)
  weights <- torch_tensor(as.matrix(dat_norm$data[, datainfo$weight_col]), device = device)
  
  phase1_variables <- torch_tensor(as.matrix(dat_norm$data[, datainfo$phase1_cols]), device = device)
  phase2_variables <- dat_norm$data[, datainfo$phase2_cols]
  phase2_variables[is.na(phase2_variables)] <- 0
  phase2_variables <- torch_tensor(as.matrix(phase2_variables), device = device)
  
  training_loss <- matrix(0, nrow = n, ncol = 3)
  
  Residual <- torch::nn_module(
    "Residual",
    initialize = function(dim1, dim2){
      self$linear <- nn_linear(dim1, dim2)
      self$bn <- nn_batch_norm1d(dim2)
      self$relu <- nn_relu()
    },
    forward = function(input){
      output <- input %>% 
        self$linear() %>%
        self$bn() %>%
        self$relu()
      return (torch_cat(list(output, input), dim = 2))
    }
  )
  
  generatorX <- torch::nn_module(
    "Generator",
    initialize = function(){
      dim1 <- 128 + cond_variables$size(2) + phase1_variables$size(2)
      dim2 <- 64
      self$seq <- torch::nn_sequential()
      for (i in 1:g_layers){
        self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
        dim1 <- dim1 + dim2
      }
      self$seq$add_module("Linear", nn_linear(dim1, length(datainfo$phase2_cols)))
    },
    forward = function(input){
      out <- self$seq(input)
      return (out)
    }
  )
  
  generatorY <- torch::nn_module(
    "Generator",
    initialize = function(){
      dim1 <- 128 + cond_variables$size(2) + phase2_variables$size(2)
      dim2 <- 64
      self$seq <- torch::nn_sequential()
      for (i in 1:g_layers){
        self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
        dim1 <- dim1 + dim2
      }
      self$seq$add_module("Linear", nn_linear(dim1, length(datainfo$phase1_cols)))
    },
    forward = function(input){
      out <- self$seq(input)
      return (out)
    }
  )
  
  discriminatorX <- torch::nn_module(
    "Discriminator",
    initialize = function(){
      self$linear1 <- nn_linear(nCol - 1, 256)
      self$linear2 <- nn_linear(256, 128)
      self$linear3 <- nn_linear(128, 1)
      self$leaky1 <- nn_leaky_relu(0.2)
      self$leaky2 <- nn_leaky_relu(0.2)
      self$dropout1 <- nn_dropout(0.5)
      self$dropout2 <- nn_dropout(0.5)
    },
    forward = function(input){
      input %>% 
        self$linear1() %>%
        self$leaky1() %>%
        self$dropout1() %>%
        self$linear2() %>%
        self$leaky2() %>%
        self$dropout2() %>%
        self$linear3()
    }
  )
  
  discriminatorY <- torch::nn_module(
    "Discriminator",
    initialize = function(){
      self$linear1 <- nn_linear(nCol - 1, 256)
      self$linear2 <- nn_linear(256, 128)
      self$linear3 <- nn_linear(128, 1)
      self$leaky1 <- nn_leaky_relu(0.2)
      self$leaky2 <- nn_leaky_relu(0.2)
      self$dropout1 <- nn_dropout(0.5)
      self$dropout2 <- nn_dropout(0.5)
    },
    forward = function(input){
      input %>% 
        self$linear1() %>%
        self$leaky1() %>%
        self$dropout1() %>%
        self$linear2() %>%
        self$leaky2() %>%
        self$dropout2() %>%
        self$linear3()
    }
  )
  
  generator_x <- generatorX()$to(device = device)
  discriminator_x <- discriminatorX()$to(device = device)
  
  generator_y <- generatorY()$to(device = device)
  discriminator_y <- discriminatorY()$to(device = device)
  
  g_solver <- torch::optim_adam(c(generator_x$parameters, generator_y$parameters), lr = lr_g, betas = c(0.5, 0.9))
  d_solver <- torch::optim_adam(c(discriminator_x$parameters, discriminator_y$parameters), lr = lr_d, betas = c(0.5, 0.9))
  
  info_list <- list(phase1_variables = phase1_variables, phase2_variables = phase2_variables, cond_variables = cond_variables, 
                    data_mask = data_mask, weights = weights, datainfo = datainfo,
                    dat_norm = dat_norm)
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss_X: :d_loss_x | D Loss_Y: :d_loss_y",
    clear = FALSE, total = n, width = 100)
  
  cross_entropy_cols <- which(!(datainfo$phase2_cols %in% datainfo$numeric_cols))
  mse_cols <- which(datainfo$phase2_cols %in% datainfo$numeric_cols)
  
  for (i in 1:n){
    batch <- samplebatches(info_list, batch_size, at_least_p = 0.25, device = device)
    X <- batch$X
    A <- batch$A
    C <- batch$C
    M <- batch$M
    
    d_solver$zero_grad()
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 128))$to(device = device)
    
    # Training Discriminator X
    fakecat_X <- torch_cat(list(fakez, A, C), dim = 2)
    fake_X <- generator_x(fakecat_X)
    
    ind_subsample <- (M[, 1] == 1)
    fake_subsample_X <- fake_X[ind_subsample, ]
    C_subsample <- C[ind_subsample, ]
    true_subsample_A <- A[ind_subsample, ]
    true_subsample_X <- X[ind_subsample, ]
    
    fakecat_sub <- torch_cat(list(fake_subsample_X, true_subsample_A, C_subsample), dim = 2)
    truecat_sub <- torch_cat(list(true_subsample_X, true_subsample_A, C_subsample), dim = 2)
    
    X_fake <- discriminator_x(fakecat_sub)
    X_true <- discriminator_x(truecat_sub)
    
    gradient_penalty <- gradient_penalty(discriminator_x, 
                                         torch_cat(list(true_subsample_X, true_subsample_A, C_subsample), dim = 2), 
                                         torch_cat(list(fake_subsample_X, true_subsample_A, C_subsample), dim = 2))
    
    dx_loss <- -(torch_mean(X_true) - torch_mean(X_fake)) + lambda * gradient_penalty
    
    # Training Discriminator Y:
    batch <- samplebatches(info_list, batch_size, at_least_p = 0.25, device = device)
    X <- batch$X
    A <- batch$A
    C <- batch$C
    M <- batch$M
    
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 128))$to(device = device)
    
    ind_subsample <- (M[, 1] == 1)
    
    fakecat_X <- torch_cat(list(fakez, A, C), dim = 2)
    fakecat_A <- torch_cat(list(fakez, X, C), dim = 2)
    fake_X <- generator_x(fakecat_X)
    fake_A <- generator_y(fakecat_A)
    fake_subsample_X <- fake_X[ind_subsample, ]
    fake_subsample_A <- fake_A[ind_subsample, ]
    
    true_subsample_X <- X[ind_subsample, ]
    true_subsample_A <- A[ind_subsample, ]
    
    fakecat_sub_A <- torch_cat(list(fake_subsample_A, true_subsample_X, C_subsample), dim = 2) #FAKE, TRUE, LABELS
    truecat_sub_A <- torch_cat(list(true_subsample_A, true_subsample_X, C_subsample), dim = 2) #TRUE, TRUE, LABELS
    
    A_fake <- discriminator_y(fakecat_sub_A)
    A_true <- discriminator_y(truecat_sub_A)
    
    gradient_penalty <- gradient_penalty(discriminator_y, 
                                         torch_cat(list(true_subsample_A, true_subsample_X, C_subsample), dim = 2), 
                                         torch_cat(list(fake_subsample_A, true_subsample_X, C_subsample), dim = 2))
    
    dy_loss <- -(torch_mean(A_true) - torch_mean(A_fake)) + lambda * gradient_penalty
    
    dx_loss$backward()
    dy_loss$backward()
    d_solver$step()
    
    # Generator X
    batch <- samplebatches(info_list, batch_size, at_least_p = 0.25, device = device)
    X <- batch$X
    A <- batch$A
    C <- batch$C
    M <- batch$M
    
    g_solver$zero_grad()
    
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 128))$to(device = device)
    fakecat_X <- torch_cat(list(fakez, A, C), dim = 2)
    fake_X <- generator_x(fakecat_X)
    
    ind_subsample <- (M[, 1] == 1)
    fake_subsample_X <- fake_X[ind_subsample, ]
    true_subsample_A <- A[ind_subsample, ]
    C_subsample <- C[ind_subsample, ]
    true_subsample_X <- X[ind_subsample, ]
    
    fakecat_sub_X <- torch_cat(list(fake_subsample_X, true_subsample_A, C_subsample), dim = 2)
    
    X_fake <- discriminator_x(fakecat_sub_X)
    g_loss_x <- -torch_mean(X_fake)
    
    if (length(mse_cols) > 0){
      mse_loss_x <- nnf_mse_loss(fake_subsample_X[, mse_cols, drop = F], true_subsample_X[, mse_cols, drop = F])
    }else{
      mse_loss_x <- 0
    }
    if (length(cross_entropy_cols) > 0){
      cross_entropy_loss_x <- nnf_binary_cross_entropy_with_logits(fake_subsample_X[, cross_entropy_cols, drop = F], 
                                                                   true_subsample_X[, cross_entropy_cols, drop = F])
    }else{
      cross_entropy_loss_x <- 0
    }
    
    # Generator Y
    fakecat_A <- torch_cat(list(fakez, fake_X, C), dim = 2) # Y is reconstructed with fake X
    fake_A <- generator_y(fakecat_A)
    fakecat_A <- torch_cat(list(fake_A, fake_X, C), dim = 2)
    fakecat_sub_A <- fakecat_A[ind_subsample, ]
    A_fake <- discriminator_y(fakecat_sub_A)
    g_loss_y <- -torch_mean(A_fake)
    
    if (length(mse_cols) > 0){
      mse_loss_y <- nnf_mse_loss(fake_A[, mse_cols, drop = F], A[, mse_cols, drop = F])
    }else{
      mse_loss_y <- 0
    }
    if (length(cross_entropy_cols) > 0){
      cross_entropy_loss_y <- nnf_binary_cross_entropy_with_logits(fake_A[, cross_entropy_cols, drop = F], 
                                                                   A[, cross_entropy_cols, drop = F])
    }else{
      cross_entropy_loss_y <- 0
    }
    
    g_loss = g_loss_x + g_loss_y + alpha * (mse_loss_x + mse_loss_y) + beta * (cross_entropy_loss_x + cross_entropy_loss_y)
    
    g_loss$backward()
    g_solver$step()
    
    training_loss[i, ] <- c(g_loss$item(), dx_loss$item(), dy_loss$item())
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", g_loss$item()),
      d_loss_x = sprintf("%.4f", dx_loss$item()),
      d_loss_y = sprintf("%.4f", dy_loss$item())
    ))
    Sys.sleep(1 / 10000)
  }
  
  
  
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "Dx Loss", "Dy Loss")
  result <- generateImpute(generator_x, m = m, dat_norm, datainfo, batch_size, device, mse_cols, cross_entropy_cols)
  
  return (list(imputation = result$imputation, loss = training_loss, sample = result$sample))
}