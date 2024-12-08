generateImpute <- function(generator, m = 5, info_list){
  list2env(info_list, envir = environment())
  
  imputed_data_list <- list()
  sample_data_list <- list()
  impute_batches <- createimputebatches(info_list)
  for (z in 1:m){
    output_tensor <- list()
    for (i in 1:ceiling(nrow(data_norm$data) / batch_size)){
      batch <- impute_batches[[i]]
      X <- batch$X
      C <- batch$C
      M <- batch$M
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      fakecat <- torch_cat(list(fakez, C), dim = 2)
      G_sample <- generator(fakecat)
      if (length(cross_entropy_cols) > 0){
        G_sample[, cross_entropy_cols] <- nnf_sigmoid(G_sample[, cross_entropy_cols])
      }
      output_tensor[[i]] <- as.matrix(G_sample$detach()$cpu())
    }
    output_tensor <- do.call(rbind, output_tensor)
    output_tensor <- cbind(output_tensor, as.matrix(weights$detach()$cpu()))
    
    sample <- denormalize(output_tensor, data_info, data_norm, method = method)
    
    data <- data_norm$data_ori
    data[is.na(data)] <- 0
    
    imputation <- as.matrix(data_mask$cpu()$detach()) * data + 
      (1 - as.matrix(data_mask$cpu()$detach())) * sample
    
    imputed_data_list[[z]] <- imputation
    sample_data_list[[z]] <- sample
  }
  return (list(imputation = imputed_data_list, sample = sample_data_list))
}

samplebatches <- function(info_list, batch_size, at_least_p = 0.2, device = "cpu"){
  list2env(info_list, envir = environment())
  ind_phase1 <- data_info$phase1_rows
  ind_phase2 <- data_info$phase2_rows
  
  sampphase2 <- sample(ind_phase2, size = as.integer(at_least_p * batch_size), 
                       prob = data_norm$data_ori[ind_phase2, data_info$weight_col] / sum(data_norm$data_ori[ind_phase2, data_info$weight_col]))
  sampphase1 <- sample(ind_phase1, size = batch_size - as.integer(at_least_p * batch_size), 
                       prob = data_norm$data_ori[ind_phase1, data_info$weight_col] / sum(data_norm$data_ori[ind_phase1, data_info$weight_col]))
  
  sampled <- c(sampphase1, sampphase2)
  sampled <- sample(sampled)
  batches <- list(X = phase2_variables[sampled, ],
                  C = phase1_variables[sampled, ],
                  M = data_mask[sampled, ],
                  W = weights[sampled, ])
  return (batches)
}

createimputebatches <- function(info_list){
  list2env(info_list, envir = environment())
  
  n <- ceiling(nrow(data_norm$data) / batch_size)
  idx <- 1
  batches <- list()
  for (i in 1:n){
    if (i == n){
      batch_size <- nrow(data_norm$data) - batch_size * (n - 1)
    }
    batches[[i]] <- list(X = phase2_variables[idx:(idx + batch_size - 1), ],
                         C = phase1_variables[idx:(idx + batch_size - 1), ],
                         M = data_mask[idx:(idx + batch_size - 1), ])
    idx <- idx + batch_size
  }
  
  return (batches = batches)
}

normalize <- function(data, data_info, method = "none"){
  if (method == "min-max"){
    maxs <- apply(data, 2, max, na.rm = T)
    mins <- apply(data, 2, min, na.rm = T)
    data_norm <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        #2 * (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6) - 1
        (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6)
        #0.5 * (1 + (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6))
      }else{
        data[, i]
      }
    }))
    
    return (list(data = data_norm,
                 maxs = maxs,
                 mins = mins,
                 data_ori = data))
  }else if (method == "zscore"){
    means <- apply(data, 2, mean, na.rm = T)
    sds <- apply(data, 2, sd, na.rm = T)
    data_norm <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        (data[, i] - means[i] + 1e-6) / (sds[i] + 1e-6)
      }else{
        data[, i]
      }
    }))
    
    return (list(data = data_norm,
                 means = means,
                 sds = sds,
                 data_ori = data))
  }else if (method == "maxoffset"){
    maxs <- apply(data, 2, max, na.rm = T)
    data_norm <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        (data[, i] + 1) / (maxs[i] + 1)
      }else{
        data[, i]
      }
    }))
    
    return (list(data = data_norm,
                 maxs = maxs,
                 data_ori = data))
  }else if (method == "none"){
    return (list(data = data, data_ori = data))
  }
}

denormalize <- function(data, data_info, params, method = "none"){
  
  if (method == "min-max"){
    maxs <- params$maxs
    mins <- params$mins
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        #((data[, i] + 1) / 2) * (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6)
        data[, i] * (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6)
        #(data[, i] * 2 - 1) * (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6)
      }else{
        round(data[, i])
      }
    }))
  }else if (method == "zscore"){
    means <- params$means
    sds <- params$sds
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        data[, i] * (sds[i] + 1e-6) + (means[i] - 1e-6)
      }else{
        round(data[, i])
      }
    }))
  }else if (method == "maxoffset"){
    maxs <- params$maxs
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        data[, i] * (maxs[i] + 1) - 1
      }else{
        round(data[, i])
      }
    }))
  }else if (method == "none"){
    data <- as.data.frame(data)
  }
  data <- as.data.frame(data)
  names(data) <- data_info$names
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

cwgangp_full <- function(data, m = 5,
                         params = list(batch_size = 256, lambda = 10, alpha = 100, n = 75, g_layers = 4, discriminator_steps = 3), 
                         sampling_info = list(phase1_cols = "X_tilde", phase2_cols = "X", weight_col = "W", 
                                              categorical_cols = c("R", "Z"), outcome_cols = c("Y")), 
                         device = "cpu",
                         norm_method = "min-max"){
  list2env(params, envir = environment())
  list2env(sampling_info, envir = environment())
  
  device <- torch_device(device)
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  new_data <- data[, c(phase2_cols, setdiff(names(data), c(phase2_cols, weight_col)), weight_col)]
  
  data_info <- getdatainfo(new_data, sampling_info)
  data_norm <- normalize(new_data, data_info, method = norm_method)
  
  data_mask <- torch_tensor(1 - is.na(data_norm$data), device = device)
  
  phase1_variables <- torch_tensor(as.matrix(data_norm$data[, -c(data_info$phase2_cols, data_info$weight_col)]), device = device)
  weights <- torch_tensor(as.matrix(data_norm$data[, data_info$weight_col]), device = device)
  
  phase2_variables <- data_norm$data[, data_info$phase2_cols]
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
      dim1 <- 256 + nCol - length(data_info$phase2_cols) - 1
      dim2 <- 64
      self$seq <- torch::nn_sequential()
      for (i in 1:g_layers){
        self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
        dim1 <- dim1 + dim2
      }
      self$seq$add_module("Linear", nn_linear(dim1, nCol - 1))
    },
    forward = function(input){
      out <- self$seq(input)
      return (out)
    }
  )
  
  discriminatorM <- torch::nn_module(
    "Discriminator",
    initialize = function(){
      dim <- nCol - 1 + phase1_variables$size()[2]
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
  
  cross_entropy_cols <- which(!((1:(nCol - 1)) %in% data_info$numeric_cols))
  mse_cols <- which((1:(nCol - 1)) %in% data_info$numeric_cols)
  
  cross_entropy_cols_c <- which(!((1:(nCol - 1))[-data_info$phase2_cols] %in% data_info$numeric_cols))
  mse_cols_c <- which((1:(nCol - 1))[-data_info$phase2_cols] %in% data_info$numeric_cols)
  
  info_list <- list(phase1_variables = phase1_variables, 
                    phase2_variables = phase2_variables,
                    data_mask = data_mask, data_info = data_info,
                    data_norm = data_norm,
                    cross_entropy_cols = cross_entropy_cols, mse_cols = mse_cols,
                    method = norm_method, weights = weights,
                    batch_size = batch_size,
                    device = device)
  epoch_result <- list()
  p <- 1
  for (i in 1:n){
    d_loss_t <- 0
    for (d in 1:discriminator_steps){
      batch <- samplebatches(info_list, at_least_p = 0.25)
      X <- batch$X
      C <- batch$C
      M <- batch$M
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
      fakecat <- torch_cat(list(fakez, C), dim = 2)
      fake <- generator(fakecat)
      
      ind_subsample <- (M[, 1] == 1)
      fake_subsample <- fake[ind_subsample, ]
      C_subsample <- C[ind_subsample, ]
      true_subsample <- torch_cat(list(X[ind_subsample, ], C_subsample), dim = 2)
      
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
    
    batch <- samplebatches(info_list, at_least_p = 0.25)
    X <- batch$X
    C <- batch$C
    M <- batch$M
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), 256))$to(device = device)
    fakecat <- torch_cat(list(fakez, C), dim = 2)
    fake <- generator(fakecat)
    
    ind_subsample <- (M[, 1] == 1)
    fake_subsample <- fake[ind_subsample, ]
    C_subsample <- C[ind_subsample, ]
    
    true_subsample <- torch_cat(list(X[ind_subsample, ], C_subsample), dim = 2)
    
    fakecat_sub <- torch_cat(list(fake_subsample, C_subsample), dim = 2)
    
    y_fake <- discriminator(fakecat_sub)
    g_loss <- -torch_mean(y_fake)
    
    if (length(mse_cols_c) > 0){
      mse_loss <- nnf_mse_loss(fake_subsample[, mse_cols_c, drop = F], true_subsample[, mse_cols_c, drop = F])
    }else{
      mse_loss <- 0
    }
    if (length(cross_entropy_cols_c) > 0){
      cross_entropy_loss <- nnf_binary_cross_entropy_with_logits(fake_subsample[, cross_entropy_cols_c, drop = F], 
                                                                 true_subsample[, cross_entropy_cols_c, drop = F])
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
      result <- generateImpute(generator, m = 1, info_list)
      epoch_result[[p]] <- result$sample
      p <- p + 1
    }
  }
  
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss")
  result <- generateImpute(generator, m = m, info_list)
  
  return (list(imputation = result$imputation, sample = result$sample, 
               epoch_result = epoch_result, loss = training_loss))
}

