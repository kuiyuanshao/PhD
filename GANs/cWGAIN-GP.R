library(torch)
library(progress)

# Function to one-hot encode categorical variables
one_hot_encode <- function(data, categorical_cols) {
  new_data <- data[, !names(data) %in% categorical_cols]
  cat_data <- data[, names(data) %in% categorical_cols]
  binary_col_indices <- list()
  binary_col_names <- list()
  for (col in categorical_cols) {
    unique_categories <- unique(na.omit(cat_data[[col]]))
    new_cols <- vector()
    for (category in unique_categories) {
      new_col_name <- paste0(col, "_", category)
      new_data[[new_col_name]] <- ifelse(cat_data[[col]] == category, 1, 0)
      new_cols <- c(new_cols, new_col_name)
    }
    binary_col_indices[[col]] <- which(names(new_data) %in% new_cols)
    binary_col_names[[col]] <- names(new_data)[binary_col_indices[[col]]]
  }
  
  return(list(data = new_data, binary_indices = binary_col_indices,
              binary_col_names = binary_col_names))
}

onehot_backtransform <- function(data, binary_indices) {
  original_data <- data[, -unlist(binary_indices)]
  for (var_name in names(binary_indices)) {
    indices <- binary_indices[[var_name]]
    binary_cols <- data[, indices, drop = FALSE]
    original_data[[var_name]] <- apply(binary_cols, 1, function(row) {
      matched <- which(row == 1)
      if (length(matched) == 1) {
        out <- sub(paste0("^", var_name, "_"), "", names(row)[matched])
        out
      } else {
        NA
      }
    })
  }
  return(original_data)
}


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
      G_sample <- activate_binary_cols(G_sample, encode_result, sampling_info$phase2_cols)
      G_sample <- torch_cat(list(G_sample, C), dim = 2)
      output_tensor[[i]] <- as.matrix(G_sample$detach()$cpu())
    }
    output_tensor <- do.call(rbind, output_tensor)
    output_tensor <- cbind(output_tensor, as.matrix(weights$detach()$cpu()))
    
    sample <- denormalize(output_tensor, data_info, data_norm, method = method)
    
    data <- data_norm$data_ori
    data[is.na(data)] <- 0
    
    imputation <- as.matrix(data_mask$cpu()$detach()) * data + 
      (1 - as.matrix(data_mask$cpu()$detach())) * sample
    
    for (i in data_info$phase2_cols){
      if (i %in% data_info$numeric_cols){
        pmm_matched <- pmm(sample[data_info$phase2_rows, i], 
                           sample[data_info$phase1_rows, i], 
                           data[data_info$phase2_rows, i], 100)
        imputation[data_info$phase1_rows, i] <- pmm_matched
      }
    }
    
    imputation <- onehot_backtransform(imputation, encode_result$binary_indices)
    sample <- onehot_backtransform(sample, encode_result$binary_indices)
    
    imputed_data_list[[z]] <- type.convert(imputation, as.is =TRUE)
    sample_data_list[[z]] <- type.convert(sample, as.is =TRUE)
  }
  return (list(imputation = imputed_data_list, sample = sample_data_list))
}

samplebatches <- function(info_list, batch_size, at_least_p = 0.2, device = "cpu"){
  list2env(info_list, envir = environment())
  ind_phase1 <- data_info$phase1_rows
  ind_phase2 <- data_info$phase2_rows
  
  #Provide a case-control based depending on phase1 and phase2 logical variables
  # phase1_binary_cols <- data_info$phase1_cols[!(data_info$phase1_cols %in% data_info$numeric_cols)]
  # phase2_binary_cols <- data_info$phase2_cols[!(data_info$phase2_cols %in% data_info$numeric_cols)]
  # #sample a binary variate to case control on in current sample
  # curr_col_1 <- sample(phase1_binary_cols, 1)
  # curr_col_2 <- sample(phase2_binary_cols, 1)
  # 
  # cases_1 <- data_norm$data[, curr_col_1, drop = F]
  # cases_2 <- data_norm$data[, curr_col_2, drop = F]
  # 
  # cases_phase1 <- cases_1[ind_phase1]
  # cases_phase2 <- cases_2[ind_phase2]
  # 
  # sampled <- c()
  # 
  # unicase <- unique(cases_phase1)
  # n_unicase <- length(unicase)
  # 
  # n1 <- batch_size - as.integer(at_least_p * batch_size)
  # n2 <- as.integer(at_least_p * batch_size)
  # 
  # n1 <- c(floor(n1 / 2), ceiling(n1 / 2))
  # n2 <- c(floor(n2 / 2), ceiling(n2 / 2))
  # 
  # for (case in 1:n_unicase){
  #   phase1 <- ind_phase1[which(cases_phase1 == unicase[case])]
  #   phase2 <- ind_phase2[which(cases_phase2 == unicase[case])]
  # 
  #   sampphase1 <- sample(phase1, size = n1[case],
  #                        prob = data_norm$data_ori[phase1, data_info$weight_col] /
  #                          sum(data_norm$data_ori[phase1, data_info$weight_col]))
  #   sampphase2 <- sample(phase2, size = n2[case],
  #                        prob = data_norm$data_ori[phase2, data_info$weight_col] /
  #                          sum(data_norm$data_ori[phase2, data_info$weight_col]))
  #   sampled <- c(sampled, sampphase1, sampphase2)
  # }
  sampphase2 <- sample(ind_phase2, size = as.integer(at_least_p * batch_size),
                      prob = data_norm$data_ori[ind_phase2, data_info$weight_col] / sum(data_norm$data_ori[ind_phase2, data_info$weight_col]))
  sampphase1 <- sample(ind_phase1, size = batch_size - as.integer(at_least_p * batch_size),
                      prob = data_norm$data_ori[ind_phase1, data_info$weight_col] / sum(data_norm$data_ori[ind_phase1, data_info$weight_col]))

  sampled <- sample(c(sampphase2, sampphase1))
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
        (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6)
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
        ifelse(data[, i] >= 0.5, 1, 0)
      }
    }))
  }else if (method == "zscore"){
    means <- params$means
    sds <- params$sds
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        data[, i] * (sds[i] + 1e-6) + (means[i] - 1e-6)
      }else{
        ifelse(data[, i] >= 0.5, 1, 0)
      }
    }))
  }else if (method == "maxoffset"){
    maxs <- params$maxs
    data <- do.call(cbind, lapply(1:ncol(data), function(i){
      if (i %in% data_info$numeric_cols){
        data[, i] * (maxs[i] + 1) - 1
      }else{
        ifelse(data[, i] >= 0.5, 1, 0)
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

pmm <- function(yhatobs, yhatmis, yobs, k) {
  idx <- mice::matchindex(d = yhatobs, t = yhatmis, k = k)
  yobs[idx]
}

cross_entropy_loss <- function(fake, true, encode_result, phase2_cols){
  cat_phase2 <- encode_result$binary_indices[which(sapply(encode_result$binary_col_names, function(col_names) {
    any(col_names %in% phase2_cols)
  }))]
  
  loss <- list()
  i <- 1
  for (cat in cat_phase2){
    loss[[i]] <- nnf_cross_entropy(fake[, cat], 
                             torch_argmax(true[, cat], dim = 2), 
                             reduction = "none")
    i <- i + 1
  }
  loss_t <- torch_stack(loss, dim = 2)$sum() / true$size(1)
  return (loss_t)
}

activate_binary_cols <- function(fake, encode_result, phase2_cols, tau = 0.2, hard = F){
  cat_phase2 <- encode_result$binary_indices[which(sapply(encode_result$binary_col_names, function(col_names) {
    any(col_names %in% phase2_cols)
  }))]
  for (cat in cat_phase2){
    fake[, cat] <- nnf_gumbel_softmax(fake[, cat], tau = tau, hard = hard)
  }
  return (fake)
}

cwgangp <- function(data, m = 5,
                    params = list(batch_size = 256, lambda = 10, alpha = 100, n = 75, g_layers = 4, discriminator_steps = 3), 
                    sampling_info = list(phase1_cols = "X_tilde", phase2_cols = "X", weight_col = "W", 
                                         categorical_cols = c("R", "Z"), outcome_cols = c("Y")), 
                    device = "cpu",
                    norm_method = "min-max"){
  list2env(params, envir = environment())
  list2env(sampling_info, envir = environment())
  
  device <- torch_device(device)

  encode_result <- one_hot_encode(data, sampling_info$categorical_cols)
  
  nRow <- dim(encode_result$data)[1]
  nCol <- dim(encode_result$data)[2]
  
  sampling_info$categorical_cols <- unlist(encode_result$binary_col_names)
  sampling_info$phase1_cols <- c(phase1_cols[!phase1_cols %in% categorical_cols], unlist(encode_result$binary_col_names[phase1_cols]))
  sampling_info$phase2_cols <- c(phase2_cols[!phase2_cols %in% categorical_cols], unlist(encode_result$binary_col_names[phase2_cols]))
  
  #reordering for concat convenience
  reordered <- encode_result$data[, c(sampling_info$phase2_cols, 
                                      setdiff(names(encode_result$data), 
                                              c(sampling_info$phase2_cols, 
                                                sampling_info$weight_col)), 
                                      sampling_info$weight_col)]
  
  binary_indices_reordered <- lapply(encode_result$binary_indices, function(indices) {
    match(names(encode_result$data)[indices], names(reordered))
  })
  
  encode_result$binary_indices <- binary_indices_reordered
  data_info <- getdatainfo(reordered, sampling_info)
  data_norm <- normalize(reordered, data_info, method = norm_method)
  
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
      self$seq$add_module("Linear", nn_linear(dim1, length(data_info$phase2_cols)))
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
      self$linear2 <- nn_linear(dim, length(data_info$phase2_cols))
      self$linear3 <- nn_linear(length(data_info$phase2_cols), 1)
      self$leaky1 <- nn_leaky_relu(0.2)
      self$leaky2 <- nn_leaky_relu(0.2)
      self$dropout1 <- nn_dropout(0.5)
      self$dropout2 <- nn_dropout(0.5)
    },
    forward = function(input){
      out <- input %>% 
        self$linear1() %>%
        self$leaky1() %>%
        self$dropout1() %>%
        self$linear2() %>%
        self$leaky2() %>%
        self$dropout2()
      self$linear3(out[, data_info$phase2_cols, drop = F])
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
  
  mse_cols <- which(data_info$phase2_cols %in% data_info$numeric_cols)
  cat_cols <- which(!data_info$phase2_cols %in% data_info$numeric_cols)
  
  info_list <- list(phase1_variables = phase1_variables, 
                    phase2_variables = phase2_variables,
                    data_mask = data_mask, data_info = data_info,
                    data_norm = data_norm, sampling_info = sampling_info,
                    encode_result = encode_result,
                    method = norm_method, weights = weights,
                    batch_size = batch_size,
                    device = device)
  epoch_result <- list()
  p <- 1
  warm_up <- 0#as.integer(0.1 * n)
  for (i in 1:n){
    d_loss_t <- 0
    if (i < warm_up){
      at_least_p <- 0.9
    }else{
      at_least_p <- 0.25
    }
    for (d in 1:discriminator_steps){
      batch <- samplebatches(info_list, at_least_p = at_least_p)
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
      
      fake_subsample <- activate_binary_cols(fake_subsample, encode_result, sampling_info$phase2_cols)
      
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
    
    batch <- samplebatches(info_list, at_least_p = at_least_p)
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
    
    fake_subsample_act <- activate_binary_cols(fake_subsample, encode_result, sampling_info$phase2_cols)
    
    fakecat_sub <- torch_cat(list(fake_subsample_act, C_subsample), dim = 2)
    
    y_fake <- discriminator(fakecat_sub)
    g_loss <- -torch_mean(y_fake)
    
    if (length(mse_cols) > 0){
      mse_loss <- nnf_mse_loss(fake_subsample[, mse_cols, drop = F], true_subsample[, mse_cols, drop = F])
    }else{
      mse_loss <- 0
    }
    if (length(cat_cols) > 0){
      ce_loss <- cross_entropy_loss(fake_subsample, true_subsample, encode_result, sampling_info$phase2_cols)
    }else{
      ce_loss <- 0
    }
    
    #ce_loss <- cross_entropy_loss(fake_subsample, true_subsample, encode_result, sampling_info$phase2_cols)
    g_loss <- gamma * g_loss + alpha * mse_loss + beta * ce_loss
    
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
  
  return (list(imputation = result$imputation, sample = result$sample, loss = training_loss,
               epoch_result = epoch_result))
}
