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

generator <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, g_dim, ncols, nphase2){
    dim1 <- g_dim[1] + ncols - nphase2
    dim2 <- g_dim[2]
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
      dim1 <- dim1 + dim2
    }
    self$seq$add_module("Linear", nn_linear(dim1, nphase2))
  },
  forward = function(input){
    out <- self$seq(input)
    return (out)
  }
)

discriminator <- torch::nn_module(
  "Discriminator",
  initialize = function(n_d_layers, ncols, pac) {
    self$pac <- pac
    self$pacdim <- ncols * pac
    self$seq <- torch::nn_sequential()
    
    dim <- self$pacdim
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
    }
    self$last_linear <- nn_linear(dim, 1)
  },
  forward = function(input) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    out <- self$last_linear(out)
    return(out)
  }
)

gradient_penalty <- function(D, real_samples, fake_samples, pac) {
  # Generate alpha for each pack (here batch_size/pac groups)
  alp <- torch_rand(c(ceiling(real_samples$size(1) / pac), 1, 1))
  alp <- alp$repeat_interleave(as.integer(pac), dim = 2)$repeat_interleave(real_samples$size(2), dim = 3)
  alp <- alp$reshape(c(-1, real_samples$size(2)))
  
  interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(TRUE)
  d_interpolates <- D(interpolates)
  
  fake <- torch_ones(d_interpolates$size())
  fake$requires_grad <- FALSE
  
  gradients <- torch::autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake,
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]
  
  # Reshape gradients to group the pac samples together
  gradients <- gradients$reshape(c(-1, pac * real_samples$size(2)))
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  
  return(gradient_penalty)
}



# discriminator <- torch::nn_module(
#   "Discriminator",
#   initialize = function(n_d_layers, ncols, nphase2){
#     dim <- ncols
#     self$seq <- torch::nn_sequential()
#     for (i in 1:n_d_layers){
#       self$seq$add_module(paste0("Linear", i), nn_linear(dim, dim))
#       self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
#       self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
#     }
#     self$last_linear <- nn_linear(dim, 1)
#   },
#   forward = function(input){
#     out <- self$seq(input)
#     out <- self$last_linear(out)
#     return (out)
#   }
# )

# gradient_penalty <- function(D, real_samples, fake_samples){
#   alp <- torch_rand(c(real_samples$size(1), 1, 1))
#   alp <- alp$repeat_interleave(real_samples$size(2), dim = 3)
#   alp <- alp$reshape(c(-1, real_samples$size(2)))
#   interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(T)
#   
#   d_interpolates <- D(interpolates)
#   fake <- torch_ones(d_interpolates$size())
#   fake$requires_grad <- F
#   gradients <- torch::autograd_grad(
#     outputs = d_interpolates,
#     inputs = interpolates,
#     grad_outputs = fake,
#     create_graph = TRUE,
#     retain_graph = TRUE
#   )[[1]]
#   gradients <- torch_reshape(gradients, c(gradients$size(1), -1))
#   gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
#   
#   return (gradient_penalty)
# }