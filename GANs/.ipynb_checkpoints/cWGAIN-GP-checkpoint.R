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
      C <- batch$C
      M <- batch$M
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      fakecat <- torch_cat(list(fakez, C), dim = 2)
      G_sample <- generator(fakecat)
      if (is.null(output_tensor)){
        if (length(cross_entropy_cols) > 0){
          binary_act <- nnf_sigmoid(G_sample[, cross_entropy_cols, drop = F])
          G_sample <- torch_cat(list(G_sample[, mse_cols, drop = F], binary_act, C), dim = 2)
        }else{
          G_sample <- torch_cat(list(G_sample, C), dim = 2)
        }
        output_tensor <- G_sample 
      }else{
        if (length(cross_entropy_cols) > 0){
          binary_act <- nnf_sigmoid(G_sample[, cross_entropy_cols, drop = F])
          G_sample <- torch_cat(list(G_sample[, mse_cols, drop = F], binary_act, C), dim = 2)
        }else{
          G_sample <- torch_cat(list(G_sample, C), dim = 2)
        }
        output_tensor <- torch_cat(list(output_tensor, G_sample), dim = 1)
      }
    }
    output_tensor <- torch_cat(list(output_tensor, impute_batches$weights), dim = 2)
    imputation <- impute_batches$data_mask * torch_cat(list(impute_batches$phase2_variables, 
                                                            impute_batches$phase1_variables,
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
                  C = phase1_variables[sampled, ],
                  M = data_mask[sampled, ],
                  W = weights[sampled, ])
  return (batches)
}

createimputebatches <- function(dat_norm, datainfo, batch_size, device = "cpu"){
  data_mask <- torch_tensor(1 - is.na(dat_norm$data), device = device)
  
  phase1_variables <- torch_tensor(as.matrix(dat_norm$data[, -c(datainfo$phase2_cols, datainfo$weight_col)]), device = device)
  weights <- torch_tensor(as.matrix(dat_norm$data[, datainfo$weight_col]), device = device)
  
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
                         C = phase1_variables[idx:(idx + batch_size - 1), ],
                         M = data_mask[idx:(idx + batch_size - 1), ])
    idx <- idx + batch_size
  }
  
  return (list(batches = batches, 
               data_mask = data_mask, 
               phase1_variables = phase1_variables, 
               phase2_variables = phase2_variables,
               weights = weights))
}

normalize <- function(data, datainfo, method = "none"){
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
  }else if (method == "zscore"){
    means <- apply(data, 2, mean, na.rm = T)
    sds <- apply(data, 2, sd, na.rm = T)
    dat_norm <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% datainfo$numeric_cols){
        (data[, i] - means[i] + 1e-6) / (sds[i] + 1e-6)
      }else{
        data[, i]
      }
    }))
    return (list(data = dat_norm,
                 means = means,
                 sds = sds,
                 data_ori = data))
  }else if (method == "maxoffset"){
    maxs <- apply(data, 2, max, na.rm = T)
    dat_norm <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% datainfo$numeric_cols){
        (data[, i] + 1) / (maxs[i] + 1)
      }else{
        data[, i]
      }
    }))
    return (list(data = dat_norm,
                 maxs = maxs,
                 data_ori = data))
  }else if (method == "none"){
    return (list(data = data, data_ori = data))
  }
}

denormalize <- function(data, datainfo, params, method = "none"){
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
  }else if (method == "zscore"){
    means <- params$means
    sds <- params$sds
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% datainfo$numeric_cols){
        data[, i] * (sds[i] + 1e-6) + (means[i] - 1e-6)
      }else{
        data[, i]
      }
    }))
    data <- as.data.frame(data)
    names(data) <- datainfo$names
  }else if (method == "maxoffset"){
    maxs <- params$maxs
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% datainfo$numeric_cols){
        data[, i] * (maxs[i] + 1) - 1
      }else{
        data[, i]
      }
    }))
    data <- as.data.frame(data)
    names(data) <- datainfo$names
  }else if (method == "none"){
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

cwgangp <- function(data, m = 5,
                    params = list(batch_size = 256, lambda = 10, alpha = 100, n = 75, g_layers = 4, discriminator_steps = 3), 
                    sampling_info = list(phase1_cols = "X_tilde", phase2_cols = "X", weight_col = "W", 
                                         categorical_cols = c("R", "Z"), outcome_cols = c("Y")), 
                    device = "cpu"){
  list2env(params, envir = environment())
  list2env(sampling_info, envir = environment())
  
  device <- torch_device(device)
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  new_data <- data[, c(phase2_cols, setdiff(names(data), c(phase2_cols, weight_col)), weight_col)]
  
  datainfo <- getdatainfo(new_data, sampling_info)
  dat_norm <- normalize(new_data, datainfo, method = "min-max")
  
  data_mask <- torch_tensor(1 - is.na(dat_norm$data), device = device)
  
  phase1_variables <- torch_tensor(as.matrix(dat_norm$data[, -c(datainfo$phase2_cols, datainfo$weight_col)]), device = device)
  weights <- torch_tensor(as.matrix(dat_norm$data[, datainfo$weight_col]), device = device)
  
  phase2_variables <- dat_norm$data[, datainfo$phase2_cols]
  phase2_variables[is.na(phase2_variables)] <- 0
  phase2_variables <- torch_tensor(as.matrix(phase2_variables), device = device)
  
  
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
  
  generatorM <- torch::nn_module(
    "Generator",
    initialize = function(){
      dim1 <- 256 + nCol - length(datainfo$phase2_cols) - 1
      dim2 <- 64
      self$seq <- torch::nn_sequential()
      for (i in 1:g_layers){
        self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
        dim1 <- dim1 + dim2
      }
      self$seq$add_module("Linear", nn_linear(dim1, length(datainfo$phase2_cols)))
      #self$seq$add_module("Tanh", nn_tanh())
    },
    forward = function(input){
      out <- self$seq(input)
      return (out)
    }
  )
  
  discriminatorM <- torch::nn_module(
    "Discriminator",
    initialize = function(){
      dim <- nCol - 1
      self$linear1 <- nn_linear(dim, dim)
      self$linear2 <- nn_linear(dim, dim)
      self$linear3 <- nn_linear(dim, 1)
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
  
  generator <- generatorM()$to(device = device)
  discriminator <- discriminatorM()$to(device = device)
  
  g_solver <- torch::optim_adam(generator$parameters, lr = lr_g, betas = c(0.5, 0.9), weight_decay = 1e-6)
  d_solver <- torch::optim_adam(discriminator$parameters, lr = lr_d, betas = c(0.5, 0.9), weight_decay = 1e-6)
  
  training_loss <- matrix(0, nrow = n, ncol = 2)
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss ",
    clear = FALSE, total = n, width = 100)
  
  info_list <- list(phase1_variables = phase1_variables, phase2_variables = phase2_variables,
                    data_mask = data_mask, weights = weights, datainfo = datainfo,
                    dat_norm = dat_norm)
  epoch_result <- list()
  p <- 1
  
  cross_entropy_cols <- which(!(datainfo$phase2_cols %in% datainfo$numeric_cols))
  mse_cols <- which(datainfo$phase2_cols %in% datainfo$numeric_cols)
  for (i in 1:n){
    d_loss_t <- 0
    for (d in 1:discriminator_steps){
      batch <- samplebatches(info_list, batch_size, at_least_p = 0.25, device = device)
      X <- batch$X
      C <- batch$C
      M <- batch$M
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      fakecat <- torch_cat(list(fakez, C), dim = 2)
      fake <- generator(fakecat)
      
      ind_subsample <- (M[, 1] == 1)
      fake_subsample <- fake[ind_subsample, ]
      C_subsample <- C[ind_subsample, ]
      true_subsample <- X[ind_subsample, ]
      
      fakecat_sub <- torch_cat(list(fake_subsample, C_subsample), dim = 2)
      truecat_sub <- torch_cat(list(true_subsample, C_subsample), dim = 2)
      
      y_fake <- discriminator(fakecat_sub)
      y_true <- discriminator(truecat_sub)
      
      gradient_penalty <- gradient_penalty(discriminator, truecat_sub, fakecat_sub)
      d_loss <- -(torch_mean(y_true) - torch_mean(y_fake))
      d_loss_gp <- d_loss + lambda * gradient_penalty
      d_loss_t <- d_loss_t + d_loss_gp$item()
      d_solver$zero_grad()
      d_loss_gp$backward()
      d_solver$step()
    }
    
    batch <- samplebatches(info_list, batch_size, at_least_p = 0.25, device = device)
    X <- batch$X
    C <- batch$C
    M <- batch$M
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
    fakecat <- torch_cat(list(fakez, C), dim = 2)
    fake <- generator(fakecat)
    
    ind_subsample <- (M[, 1] == 1)
    fake_subsample <- fake[ind_subsample, ]
    C_subsample <- C[ind_subsample, ]
    true_subsample <- X[ind_subsample, ]
    
    fakecat_sub <- torch_cat(list(fake_subsample, C_subsample), dim = 2)
    
    y_fake <- discriminator(fakecat_sub)
    g_loss <- -torch_mean(y_fake)
    
    if (length(mse_cols) > 0){
      mse_loss <- nnf_mse_loss(fake_subsample[, mse_cols, drop = F], true_subsample[, mse_cols, drop = F])
    }else{
      mse_loss <- 0
    }
    if (length(cross_entropy_cols) > 0){
      cross_entropy_loss <- nnf_binary_cross_entropy_with_logits(fake_subsample[, cross_entropy_cols, drop = F], 
                                                                 true_subsample[, cross_entropy_cols, drop = F])
    }else{
      cross_entropy_loss <- 0
    }
    g_loss <- gamma * g_loss + alpha * mse_loss + beta * cross_entropy_loss
    
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
    
    training_loss[i, ] <- c(g_loss$item(), d_loss_t / discriminator_steps)
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", g_loss$item()),
      d_loss = sprintf("%.4f", d_loss_t / discriminator_steps)
    ))
    Sys.sleep(1 / 10000)
    
    if (i %% 100 == 0){
      result <- generateImpute(generator, m = 1, dat_norm, datainfo, batch_size, device, mse_cols, cross_entropy_cols)
      epoch_result[[p]] <- result$sample
      p <- p + 1
    }
  }
  
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss")
  result <- generateImpute(generator, m = m, dat_norm, datainfo, batch_size, device, mse_cols, cross_entropy_cols)
  
  return (list(imputation = result$imputation, loss = training_loss, sample = result$sample,
               epoch_result = epoch_result))
}



cwgangp_full <- function(data, m = 5,
                         params = list(batch_size = 256, lambda = 10, alpha = 100, n = 100), 
                         sampling_info = list(phase1_cols = "X_tilde", phase2_cols = "X", weight_col = "W", categorical_cols), 
                         device = "cpu"){
  list2env(params, envir = environment())
  list2env(sampling_info, envir = environment())
  
  device <- torch_device(device)
  
  new_data <- data
  phase1_rows <- as.numeric(is.na(rowSums(new_data)))
  phase2_rows <- as.numeric(!is.na(rowSums(new_data)))
  nRow <- dim(new_data)[1]
  nCol <- dim(new_data)[2]
  
  new_data <- new_data[, c(phase2_cols, setdiff(names(new_data), phase2_cols))]
  maxs <- apply(new_data, 2, max, na.rm = T)
  mins <- apply(new_data, 2, min, na.rm = T)
  
  phase1_cols <- which(names(new_data) %in% phase1_cols)
  phase2_cols <- which(names(new_data) %in% phase2_cols)
  numeric_cols <- which(!(names(new_data) %in% categorical_cols))
  data_norm <- do.call(cbind, lapply(1:nCol, function(i){
    if (i %in% numeric_cols){
      (new_data[, i] - mins[i] + 1) / (maxs[i] - mins[i] + 1) #adding a small positive value since missing values will be 0
    }else{
      new_data[, i]
    }
  }))
  
  data_mask <- torch_tensor(1 - is.na(new_data), device = device)
  data_phase1 <- torch_tensor(as.matrix(data_norm[, -phase2_cols]), device = device)
  data_error <- torch_tensor(as.matrix(data_norm[, phase1_cols]), device = device)
  phase2_vals <- data_norm[, phase2_cols]
  phase2_vals[is.na(phase2_vals)] <- 0
  data_phase2 <- torch_tensor(as.matrix(phase2_vals), device = device)
  
  createtorchdata <- dataset(
    name = "torch data",
    initialize = function(phase2samples, phase1samples, mask, errorprone){
      self$X <- phase2samples
      self$C <- phase1samples
      self$M <- mask
      self$A <- errorprone
    },
    .getitem = function(index){
      list(X = self$X[index, ], C = self$C[index, ], 
           M = self$M[index, ], A = self$A[index, ])
    },
    .length = function(){
      self$M$size(1)
    }
  )
  
  dataset <- createtorchdata(data_phase2, data_phase1, data_mask, data_error)
  
  dataloader <- dataloader(
    dataset = dataset,
    batch_size = batch_size,   # Size of each batch
    shuffle = F     # Shuffle data at the start of each epoch
  )
  
  training_loss <- matrix(0, nrow = n, ncol = 2)
  
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
  
  GAIN_Generator <- torch::nn_module(
    "Generator",
    initialize = function(){
      dim1 <- 256 + data_phase1$size(2)
      dim2 <- 64
      self$seq <- torch::nn_sequential()
      for (i in 1:3){
        self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
        dim1 <- dim1 + dim2
      }
      self$seq$add_module("Linear", nn_linear(dim1, nCol))
      #self$seq$add_module("Sigmoid", nn_sigmoid()) #To limit the output between 0 and 1 #But performed really bad
    },
    forward = function(input){
      out <- self$seq(input)
      return (out)
    }
  )
  
  GAIN_Discriminator <- torch::nn_module(
    "Discriminator",
    initialize = function(){
      self$linear1 <- nn_linear(nCol + data_phase1$size(2), 256)
      self$linear2 <- nn_linear(256, 256)
      self$linear3 <- nn_linear(256, 1)
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
  
  G_layer <- GAIN_Generator()$to(device = device)
  D_layer <- GAIN_Discriminator()$to(device = device)
  
  G_solver <- torch::optim_adam(G_layer$parameters, lr = 1e-4, betas = c(0, 0.9))
  D_solver <- torch::optim_adam(D_layer$parameters, lr = 1e-4, betas = c(0, 0.9))
  
  generator <- function(X, C){
    input <- torch_cat(list(X, C), dim = 2)
    return (G_layer(input))
  }
  
  discriminator <- function(X, C){
    input <- torch_cat(list(X, C), dim = 2)
    return (D_layer(input))
  }
  
  
  gradient_penalty <- function(D, real_samples, fake_samples, C, M){
    alp <- torch_rand(c(real_samples$size(1), 1, 1))
    alp <- alp$repeat_interleave(real_samples$size(2), dim = 3)
    alp <- alp$reshape(c(-1, real_samples$size(2)))
    interpolates <- (alp * (M * real_samples) + (1 - alp) * ((1 - M) * fake_samples))$requires_grad_(T)
    d_interpolates <- D(interpolates, C)
    fake <- torch_full(c(real_samples$size(1), 1), 1.0)
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
  
  G_loss <- function(N, X, C, M){
    fake <- generator(N, C)
    true <- torch_cat(list(X, C), dim = 2)
    y_fake <- discriminator(fake, C)
    g_loss <- -torch_mean((1 - M[, 1, drop = FALSE]) * y_fake)
    
    mse_loss <- torch_mean((M * fake - M * true) ^ 2) / torch_mean(M)
    G_loss <- g_loss + alpha * mse_loss
    return (G_loss$to(device = device))
  }
  D_loss <- function(N, X, C, M){
    fake <- generator(N, C)
    true <- torch_cat(list(X, C), dim = 2)
    y_fake <- discriminator(fake, C)
    y_true <- discriminator(true, C)
    
    gradient_penalty <- gradient_penalty(discriminator, true, fake, C, M)
    D_loss <- -(torch_mean(M[, 1, drop = FALSE] * y_true) - 
                  torch_mean((1 - M[, 1, drop = FALSE]) * y_fake)) + 
      lambda * gradient_penalty
    return (D_loss$to(device = device))
  }
  
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss",
    clear = FALSE, total = n, width = 100)
  
  for (i in 1:n){
    g_loss_t <- 0
    d_loss_t <- 0
    coro::loop(for (batch in dataloader){
      X <- batch$X
      A <- batch$A
      C <- batch$C
      M <- batch$M
      
      if (i %% i == 0) {
        #fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
        #fakez <- torch_normal(mean = 0, std = 1, 
        #                      size = c(X$size(1), 
        #                               as.integer(length(phase2_cols))))$to(device = device)
        #fakez <- M[, 1, drop = F] * X + (1 - M[, 1, drop = F]) * fakez
        
        fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
        
        d_loss <- D_loss(fakez, X, C, M)
        
        D_solver$zero_grad()
        d_loss$backward()
        D_solver$step()
        d_loss_t <- d_loss_t + d_loss$item()
      }
      
      #fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      g_loss <- G_loss(fakez, X, C, M)
      
      G_solver$zero_grad()
      g_loss$backward()
      G_solver$step()
      
      g_loss_t <- g_loss_t + g_loss$item()
    })
    training_loss[i, ] <- c(g_loss_t / length(dataloader), d_loss_t / length(dataloader))
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", g_loss_t / length(dataloader)),
      d_loss = sprintf("%.4f", d_loss_t / length(dataloader))
    ))
    Sys.sleep(1 / 10000)
  }
  imputed_data_list <- list()
  sample_data_list <- list()
  for (z in 1:m){
    output_tensor <- NULL
    coro::loop(for (batch in dataloader){
      X <- batch$X
      C <- batch$C
      M <- batch$M
      A <- batch$A
      #fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      G_sample <- generator(fakez, C)
      if (is.null(output_tensor)){
        output_tensor <- G_sample 
      }else{
        output_tensor <- torch_cat(list(output_tensor, G_sample), dim = 1)
      }
    })
    imputation <- dataset$M * torch_cat(list(dataset$X, dataset$C), dim = 2) + (1 - dataset$M) * output_tensor
    imputation <- as.matrix(imputation$detach()$cpu())
    imputation <- do.call(cbind, lapply(1:nCol, function(i){
      if (i %in% numeric_cols){
        (imputation[, i]) * (maxs[i] - mins[i] + 1) + (mins[i] - 1)
      }else{
        imputation[, i]
      }
    }))
    
    sample <- as.matrix(output_tensor$detach()$cpu())
    sample <- do.call(cbind, lapply(1:nCol, function(i){
      if (i %in% numeric_cols){
        (sample[, i]) * (maxs[i] - mins[i] + 1) + (mins[i] - 1)
      }else{
        sample[, i]
      }
    }))
    
    imputation <- data.frame(as.matrix(imputation))
    names(imputation) <- names(new_data)
    sample <- data.frame(as.matrix(sample))
    names(sample) <- names(new_data)
    
    imputed_data_list[[z]] <- imputation
    sample_data_list[[z]] <- sample
  }
  
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss")
  
  return (list(imputation = imputed_data_list, loss = training_loss, sample = sample_data_list))
}

cwgangp_mask <- function(data, m = 5,
                         params = list(batch_size = 256, lambda = 10, alpha = 100, n = 100), 
                         sampling_info = list(phase1_cols = "X_tilde", phase2_cols = "X", weight_col = "W", categorical_cols), 
                         device = "cpu"){
  list2env(params, envir = environment())
  list2env(sampling_info, envir = environment())
  
  device <- torch_device(device)
  
  new_data <- data
  phase1_rows <- as.numeric(is.na(rowSums(new_data)))
  phase2_rows <- as.numeric(!is.na(rowSums(new_data)))
  nRow <- dim(new_data)[1]
  nCol <- dim(new_data)[2]
  
  new_data <- new_data[, c(phase2_cols, setdiff(names(new_data), phase2_cols))]
  maxs <- apply(new_data, 2, max, na.rm = T)
  mins <- apply(new_data, 2, min, na.rm = T)
  
  phase1_cols <- which(names(new_data) %in% phase1_cols)
  phase2_cols <- which(names(new_data) %in% phase2_cols)
  numeric_cols <- which(!(names(new_data) %in% categorical_cols))
  data_norm <- do.call(cbind, lapply(1:nCol, function(i){
    if (i %in% numeric_cols){
      (new_data[, i] - mins[i] + 1) / (maxs[i] - mins[i] + 1) #adding a small positive value since missing values will be 0
    }else{
      new_data[, i]
    }
  }))
  
  data_mask <- torch_tensor(1 - is.na(new_data), device = device)
  data_phase1 <- torch_tensor(as.matrix(data_norm[, -phase2_cols]), device = device)
  data_error <- torch_tensor(as.matrix(data_norm[, phase1_cols]), device = device)
  phase2_vals <- data_norm[, phase2_cols]
  phase2_vals[is.na(phase2_vals)] <- 0
  data_phase2 <- torch_tensor(as.matrix(phase2_vals), device = device)
  
  createtorchdata <- dataset(
    name = "torch data",
    initialize = function(phase2samples, phase1samples, mask, errorprone){
      self$X <- phase2samples
      self$C <- phase1samples
      self$M <- mask
      self$A <- errorprone
    },
    .getitem = function(index){
      list(X = self$X[index, ], C = self$C[index, ], 
           M = self$M[index, ], A = self$A[index, ])
    },
    .length = function(){
      self$M$size(1)
    }
  )
  
  dataset <- createtorchdata(data_phase2, data_phase1, data_mask, data_error)
  
  dataloader <- dataloader(
    dataset = dataset,
    batch_size = batch_size,   # Size of each batch
    shuffle = F     # Shuffle data at the start of each epoch
  )
  
  training_loss <- matrix(0, nrow = n, ncol = 2)
  
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
  
  GAIN_Generator <- torch::nn_module(
    "Generator",
    initialize = function(){
      dim1 <- 256 + data_phase1$size(2)
      dim2 <- 64
      self$seq <- torch::nn_sequential()
      for (i in 1:4){
        self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
        dim1 <- dim1 + dim2
      }
      self$seq$add_module("Linear", nn_linear(dim1, length(phase2_cols)))
      #self$seq$add_module("Sigmoid", nn_sigmoid()) #To limit the output between 0 and 1 #But performed really bad
    },
    forward = function(input){
      out <- self$seq(input)
      return (out)
    }
  )
  
  GAIN_Discriminator <- torch::nn_module(
    "Discriminator",
    initialize = function(){
      self$linear1 <- nn_linear(nCol, 256)
      self$linear2 <- nn_linear(256, 256)
      self$linear3 <- nn_linear(256, 1)
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
  
  G_layer <- GAIN_Generator()$to(device = device)
  D_layer <- GAIN_Discriminator()$to(device = device)
  
  G_solver <- torch::optim_adam(G_layer$parameters, lr = 1e-4, betas = c(0, 0.9))
  D_solver <- torch::optim_adam(D_layer$parameters, lr = 1e-4, betas = c(0, 0.9))
  
  generator <- function(X, C){
    input <- torch_cat(list(X, C), dim = 2)
    return (G_layer(input))
  }
  
  discriminator <- function(X, C){
    input <- torch_cat(list(X, C), dim = 2)
    return (D_layer(input))
  }
  
  
  gradient_penalty <- function(D, real_samples, fake_samples, C, M){
    alp <- torch_rand(c(real_samples$size(1), 1, 1))
    alp <- alp$repeat_interleave(real_samples$size(2), dim = 3)
    alp <- alp$reshape(c(-1, real_samples$size(2)))
    interpolates <- (alp * (M * real_samples) + (1 - alp) * ((1 - M) * fake_samples))$requires_grad_(T)
    d_interpolates <- D(interpolates, C)
    fake <- torch_full(c(real_samples$size(1), 1), 1.0)
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
  
  G_loss <- function(N, X, C, M){
    fake <- generator(N, C)
    #true <- torch_cat(list(X, C), dim = 2)
    y_fake <- discriminator(fake, C)
    g_loss <- -torch_mean((1 - M[, 1, drop = FALSE]) * y_fake)
    
    mse_loss <- torch_mean((M[, 1, drop = FALSE] * fake -  
                              M[, 1, drop = FALSE] * X) ^ 2) / 
      torch_mean(M[, 1, drop = FALSE])
    G_loss <- g_loss + alpha * mse_loss
    return (G_loss$to(device = device))
  }
  D_loss <- function(N, X, C, M){
    fake <- generator(N, C)
    #true <- torch_cat(list(X, C), dim = 2)
    y_fake <- discriminator(fake, C)
    y_true <- discriminator(X, C)
    
    gradient_penalty <- gradient_penalty(discriminator, X, fake, C, M[, 1, drop = FALSE])
    D_loss <- -(torch_mean(M[, 1, drop = FALSE] * y_true) - 
                  torch_mean((1 - M[, 1, drop = FALSE]) * y_fake)) + 
      lambda * gradient_penalty
    return (D_loss$to(device = device))
  }
  
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss",
    clear = FALSE, total = n, width = 100)
  
  for (i in 1:n){
    g_loss_t <- 0
    d_loss_t <- 0
    coro::loop(for (batch in dataloader){
      X <- batch$X
      A <- batch$A
      C <- batch$C
      M <- batch$M
      
      if (i %% i == 0) {
        #fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
        #fakez <- torch_normal(mean = 0, std = 1, 
        #                      size = c(X$size(1), 
        #                               as.integer(length(phase2_cols))))$to(device = device)
        #fakez <- M[, 1, drop = F] * X + (1 - M[, 1, drop = F]) * fakez
        
        fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
        
        d_loss <- D_loss(fakez, X, C, M)
        
        D_solver$zero_grad()
        d_loss$backward()
        D_solver$step()
        d_loss_t <- d_loss_t + d_loss$item()
      }
      
      #fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      g_loss <- G_loss(fakez, X, C, M)
      
      G_solver$zero_grad()
      g_loss$backward()
      G_solver$step()
      
      g_loss_t <- g_loss_t + g_loss$item()
    })
    training_loss[i, ] <- c(g_loss_t / length(dataloader), d_loss_t / length(dataloader))
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", g_loss_t / length(dataloader)),
      d_loss = sprintf("%.4f", d_loss_t / length(dataloader))
    ))
    Sys.sleep(1 / 10000)
  }
  
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss")
  
  
  return (list(imputation = imputed_data_list, loss = training_loss, sample = sample_data_list))
}



