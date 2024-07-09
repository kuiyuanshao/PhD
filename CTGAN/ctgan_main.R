library(caret)
library(torch)
library(progress)
library(reticulate)
library(tidyverse)


ctgan <- function(data, device = "cpu", batch_size = 500, embedding_dim = 128, 
                  generator_dim = c(256, 256), discriminator_dim = c(256, 256), 
                  generator_lr = 2e-4, generator_decay = 1e-6, 
                  discriminator_lr = 2e-4, discriminator_decay = 1e-6, 
                  discriminator_steps = 1, log_frequency = T, epochs = 300, pac = 10){
  source_python("data_sampler.py")
  source_python("data_transformer.py")
  device <- torch_device(device)
  
  Residual <- torch::nn_module(
    "Residual",
    initialize = function(i, o){
      self$fc <- torch::nn_linear(i, o)
      self$bn <- torch::nn_batch_norm1d(o)
      self$relu <- torch::nn_relu()
    },
    forward = function(input){
      out <- self$fc(input)
      out <- self$bn(out)
      out <- self$relu(out)
      return (torch_cat(c(out, input), dim = 2))
    }
  )
  
  Generator <- torch::nn_module(
    "Generator",
    initialize = function(em_dim, g_dim, data_dim){
      dim <- em_dim
      self$seq <- torch::nn_sequential()
      i <- 1
      for (item in g_dim){
        self$seq$add_module(Residual(dim, item), name = paste0("Module_", i))
        dim <- dim + item
        i <- i + 1
      }
      self$seq$add_module(nn_linear(dim, data_dim), name = "LastLinear_G")
    },
    forward = function(input){
      data <- self$seq(input)
      return (data)
    }
    
  )
  
  Discriminator <- torch::nn_module(
    "Discriminator",
    initialize = function(input_dim, d_dim, pac = 10){
      dim <- input_dim * pac
      self$pac <- pac
      self$pacdim <- dim
      self$seq <- torch::nn_sequential()
      i <- 1
      for (item in d_dim){
        self$seq$add_module(nn_linear(dim, item), name = paste0("Linear_", i))
        self$seq$add_module(nn_leaky_relu(0.2), name = paste0("Activation_", i))
        self$seq$add_module(nn_dropout(0.5), name = paste0("Dropout_", i))
        i <- i + 1
        dim <- item
      }
      self$seq$add_module(nn_linear(dim, 1), name = "LastLinear_D")
    },
    cal_gp = function(real_data, fake_data, device = "cpu", pac = 10, lambda = 10){
      alpha <- torch_rand(c(real_data$size(1) %/% pac, 1, 1), device = device)
      alpha <- alpha$`repeat`(c(1, pac, real_data$size(2)))
      alpha <- alpha$view(c(-1, real_data$size(2)))
      
      interpolated <- (alpha * real_data + (1 - alpha) * fake_data)
      #interpolated <- Variable(interpolated, requires_grad = T)
      prob <- self$forward(interpolated)
      
      gradients <- autograd_grad(outputs = prob, inputs = interpolated,
                                 grad_outputs = torch_ones(prob$size(), device = device),
                                 create_graph = T,
                                 retain_graph = T)[[1]]
      gradients <- gradients$view(c(-1, pac * real_data$size(2)))$norm(2, dim = 2) - 1
      
      #print((gradients ^ 2)$mean() * 10)
      grad_penalty <- (gradients ^ 2)$mean() * lambda
      
      return (grad_penalty)
    },
    forward = function(input){
      return (self$seq(input$view(c(-1, self$pacdim))))
    }
  )
  
  Residual <- torch::nn_module(
    "Residual",
    initialize = function(i, o){
      self$fc <- torch::nn_linear(i, o)
      self$bn <- torch::nn_batch_norm1d(o)
      self$relu <- torch::nn_relu()
    },
    forward = function(input){
      out <- self$fc(input)
      out <- self$bn(out)
      out <- self$relu(out)
      return (torch_cat(c(out, input), dim = 2))
    }
  )
  
  Generator <- torch::nn_module(
    "Generator",
    initialize = function(em_dim, g_dim, data_dim){
      dim <- em_dim
      self$seq <- torch::nn_sequential()
      i <- 1
      for (item in g_dim){
        self$seq$add_module(Residual(dim, item), name = paste0("module", i))
        dim <- dim + item
        i <- i + 1
      }
      self$seq$add_module(nn_linear(dim, data_dim), name = "lastlinear1")
    },
    forward = function(input){
      data <- self$seq(input)
      return (data)
    }
    
  )
  
  Discriminator <- torch::nn_module(
    "Discriminator",
    initialize = function(input_dim, d_dim, pac = 10){
      dim <- input_dim * pac
      self$pac <- pac
      self$pacdim <- dim
      self$seq <- torch::nn_sequential()
      i <- 1
      for (item in d_dim){
        self$seq$add_module(nn_linear(dim, item), name = paste0("linear", i))
        self$seq$add_module(nn_leaky_relu(0.2), name = paste0("activation", i))
        self$seq$add_module(nn_dropout(0.5), name = paste0("dropout", i))
        i <- i + 1
        dim <- item
      }
      self$seq$add_module(nn_linear(dim, 1), name = "lastlinear2")
    },
    cal_gp = function(real_data, fake_data, device = "cpu", pac = 10, lambda = 10){
      alpha <- torch_rand(c(real_data$size(1) %/% pac, 1, 1), device = device)
      alpha <- alpha$`repeat`(c(1, pac, real_data$size(2)))
      alpha <- alpha$view(c(-1, real_data$size(2)))
      
      interpolated <- (alpha * real_data + (1 - alpha) * fake_data)
      #interpolated <- Variable(interpolated, requires_grad = T)
      prob <- self$forward(interpolated)
      
      gradients <- autograd_grad(outputs = prob, inputs = interpolated,
                                 grad_outputs = torch_ones(prob$size(), device = device),
                                 create_graph = T,
                                 retain_graph = T)[[1]]
      gradients <- gradients$view(c(-1, pac * real_data$size(2)))$norm(2, dim = 2) - 1
      
      #print((gradients ^ 2)$mean() * 10)
      grad_penalty <- (gradients ^ 2)$mean() * lambda
      
      return (grad_penalty)
    },
    forward = function(input){
      return (self$seq(input$view(c(-1, self$pacdim))))
    }
  )
  
  
  discrete_columns <- list("Z")
  
  transformer <- DataTransformer()
  transformer$fit(data, discrete_columns)
  train_data <- transformer$transform(data)
  
  data_sampler <- DataSampler(train_data, transformer$output_info_list, log_frequency)
  
  data_dim <- transformer$output_dimensions
  
  generator <- Generator(embedding_dim + data_sampler$dim_cond_vec(), 
                         generator_dim, data_dim)$to(device = device)
  discriminator <- Discriminator(data_dim + data_sampler$dim_cond_vec(), 
                                 discriminator_dim, pac = pac)$to(device = device)
  
  optimizerG <- optim_adam(generator$parameters, 
                           lr = generator_lr, 
                           betas = c(0.5, 0.9), 
                           weight_decay = generator_decay)
  optimizerD <- optim_adam(discriminator$parameters, 
                           lr = discriminator_lr, 
                           betas = c(0.5, 0.9), 
                           weight_decay = discriminator_decay)
  
  mean <- torch_zeros(batch_size, embedding_dim, device = device)
  std <- mean + 1
  
  steps_per_epoch <- max(dim(train_data)[1] %/% batch_size, 1)
  
  loss_values <- matrix(0, ncol = 3, 
                        nrow = epochs * steps_per_epoch)
  k <- 1
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = epochs * steps_per_epoch * discriminator_steps, width = 60)
  for (i in 1:epochs){
    for (id in 1:steps_per_epoch){
      for (n in 1:discriminator_steps){
        fakez <- torch_normal(mean = mean, std = std)
        condvec <- data_sampler$sample_condvec(as.integer(batch_size))

        if (is.null(condvec)){
          c1 <- m1 <- col <- opt <- NULL
          real <- data_sampler$sample_data(train_data, as.integer(batch_size), col, opt)
        }else{
          c1 <- condvec[[1]]
          m1 <- condvec[[2]]
          col <- condvec[[3]]
          opt <- condvec[[4]]
          c1 <- torch_tensor(c1, device = device)
          m1 <- torch_tensor(m1, device = device)
          fakez <- torch_cat(list(fakez, c1), dim = 2)
          
          perm <- sample(1:batch_size)
          real <- data_sampler$sample_data(train_data, as.integer(batch_size), 
                                           as.integer(col[perm]), as.integer(opt[perm]))
          c2 <- c1[perm]
        }
        fake <- generator(fakez)
        
        
        fakeact <- apply_activate(fake, transformer$output_info_list, device = device)
        real <- torch_tensor(real, device = device, dtype = torch_float32())
        
        if (!is.null(c1)){
          fake_cat <- torch_cat(list(fakeact, c1), dim = 2)
          real_cat <- torch_cat(list(real, c2), dim = 2)
        }else{
          real_cat <- real
          fake_cat <- fakeact
        }
        
        y_fake <- discriminator(fake_cat)
        y_real <- discriminator(real_cat)
        
        pen <- discriminator$cal_gp(real_cat, fake_cat, device = device, pac = pac)
        
        loss_d <- -(torch_mean(y_real) - torch_mean(y_fake))
        
        optimizerD$zero_grad()
        pen$backward(retain_graph = T)
        loss_d$backward()
        optimizerD$step()
      }
      fakez <- torch_normal(mean = mean, std = std)
      condvec <- data_sampler$sample_condvec(as.integer(batch_size))
      
      if (is.null(condvec)){
        c1 <- m1 <- col <- opt <- NULL
      }else{
        c1 <- condvec[[1]]
        m1 <- condvec[[2]]
        col <- condvec[[3]]
        opt <- condvec[[4]]
        c1 <- torch_tensor(c1, device = device)
        m1 <- torch_tensor(m1, device = device)
        fakez <- torch_cat(list(fakez, c1), dim = 2)
      }
      fake <- generator(fakez)
      fakeact <- apply_activate(fake, transformer$output_info_list, device = device)
      if (!is.null(c1)){
        y_fake <- discriminator(torch_cat(list(fakeact, c1), dim = 2))
      }else{
        y_fake <- discriminator(fakeact)
      }
      
      if (is.null(condvec)){
        cross_entropy <- 0
      }else{
        cross_entropy <- cond_loss(fake, c1, m1, transformer$output_info_list)
      }
      
      loss_g <- -torch_mean(y_fake) + cross_entropy
      #print(loss_g)
      
      optimizerG$zero_grad()
      loss_g$backward()
      optimizerG$step()
      
      generator_loss <- torch_mean(loss_g)$detach()$cpu()
      discriminator_loss <- torch_mean(loss_d)$detach()$cpu()
      
      loss_values[k, ] <- c(k, as.numeric(generator_loss), 
                            as.numeric(discriminator_loss))
      k <- k + 1
      
      pb$tick(tokens = list(what = "CTGAN   "))
      Sys.sleep(1 / 10000)
    }
    
  }
  sampled <- random_samp(dim(data)[1],
                         transformer = transformer, 
                         data_sampler = data_sampler, 
                         batch_size = batch_size, 
                         embedding_dim = embedding_dim,
                         generator = generator,
                         device = device)
  
  return (list(loss_values, sampled))
  
}

random_samp <- function(n, condition_column = NULL, condition_value = NULL, 
                        transformer, data_sampler, batch_size, embedding_dim, generator, device){
  if (!is.null(condition_column) & !is.null(condition_value)){
    condition_info <- transformer$convert_column_name_value_to_id(condition_column, condition_value)
    global_condition_vec <- data_sampler$generate_cond_from_condition_column_info(condition_info, batch_size)
  }else{
    global_conidtion_vec <- NULL
  }
  
  steps <- n %/% batch_size
  data <- list()
  data_j <- list()
  for (i in 1:steps){
    mean <- torch_zeros(batch_size, embedding_dim)
    std <- mean + 1
    fakez <- torch_normal(mean = mean, std = std)$to(device = device)
    
    if (!is.null(global_conidtion_vec)){
      condvec <- global_condition_vec
    }else{
      condvec <- data_sampler$sample_original_condvec(as.integer(batch_size))
    }
    
    if (is.null(condvec)){
    }else{
      c1 <- condvec
      c1 <- torch_tensor(c1, device = device)
      fakez <- torch_cat(list(fakez, c1), dim = 2)
    }
    
    fake <- generator(fakez)
    fakeact <- apply_activate(fake, transformer$output_info_list, device = device)
    data[[i]] <- data.frame(as.matrix(fakeact$detach()$cpu()))
  }
  data <- bind_rows(data)
  return (transformer$inverse_transform(as.matrix(data)))
}

apply_activate <- function(data, output_info_list, device){
  data_t <- list()
  st <- 1
  i <- 1
  for (column_info in output_info_list){
    for (span_info in column_info){
      if (span_info$activation_fn == 'tanh'){
        data_t[[i]] <- torch_tanh(as.matrix(data[, st]$detach()$cpu()))
        st <- st + span_info$dim
        i <- i + 1
      }else if (span_info$activation_fn == 'softmax'){
        ed <- st + span_info$dim
        transformed <- nnf_gumbel_softmax(data[, st:(ed-1)], tau = 0.2, hard = F, dim = -1)
        data_t[[i]] <- transformed
        st <- ed
        i  <- i + 1
      }
    }
  }
  return (torch_cat(data_t, dim = 2))
}

gumbel_softmax <- function(data, tau){
  for (i in 1:10){
    transformed <- nnf_gumbel_softmax(data, tau)
    #print(transformed)
    if (!(as.logical(torch_isnan(transformed)$any()$detach()$cpu()))){
      return (transformed)
    }
  }
}


cond_loss <- function(data, c, m, output_info_list){
  loss <- list()
  st <- 1
  st_c <- 1
  i <- 1
  for (column_info in output_info_list){
    for (span_info in column_info){
      if (length(column_info) != 1 | span_info$activation_fn != 'softmax'){
        st <- st + span_info$dim
      }else{
        ed <- st + span_info$dim
        ed_c <- st_c + span_info$dim
        tmp <- nnf_cross_entropy(data[, st:(ed - 1)], 
                                 torch_argmax(c[, st_c:(ed_c - 1)], dim = 2), 
                                 reduction = "none")
        loss[[i]] <- tmp
        st <- ed
        st_c <- ed_c
        i <- i + 1
      }
    }
  }
  loss <- torch_stack(loss, dim = 2)

  return ((loss * m)$sum() / data$size()[1])
}

simtwophasepaper <- function(SD = c(sqrt(3), sqrt(3))){
  n  <- 1000
  beta <- c(1, 1, 1)
  e_U <- SD
  mx <- 0; sx <- 1; zrange <- 1; zprob <- .5
  simZ   <- rbinom(n, zrange, zprob)
  simX   <- (1-simZ)*rnorm(n, 0, 1) + simZ*rnorm(n, 0.5, 1)
  epsilon <- rnorm(n, 0, 1)
  simY    <- beta[1] + beta[2]*simX + beta[3]*simZ + epsilon
  simX_tilde <- simX + rnorm(n, 0, e_U[1]*(simZ==0) + e_U[2]*(simZ==1))
  data <- data.frame(Y_tilde=simY, X_tilde=simX_tilde, Y=simY, X=simX, Z=simZ)
  return (data)
}
