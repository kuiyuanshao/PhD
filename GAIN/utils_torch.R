
onehot_encoding <- function(data, ind){
  one_hot_data <- data
  if (length(ind) == 1){
    one_hot_data[, ind] <- as.character(one_hot_data[, ind])
    unique_category <- unique(data[, ind])
    no_cat <- length(unique_category[!is.na(unique_category)])
  }else{
    one_hot_data[, ind] <- lapply(one_hot_data[, ind], as.character)
    unique_category <- lapply(data[, ind], unique)
    no_cat <- unlist(lapply(unique_category, function(x){
                                  length(x[!is.na(x)])}))
  }
  one_hot <- caret::dummyVars(" ~ .", data = one_hot_data)
  one_hot_data <- data.frame(predict(one_hot, newdata = one_hot_data))
  
  new_ind <- c(ind[1], ind + no_cat)
  cutpoint <- new_ind[-length(new_ind)]
  
  new_ind <- which(!(colnames(one_hot_data) %in% colnames(data)))
  
  count <- 1
  for (i in new_ind){
    if (i %in% cutpoint[-1]){
      count <- count + 1
    }
    ind_inactive <- which(one_hot_data[, i] == 0)
    one_hot_data[ind_inactive, i] <- runif(length(ind_inactive), 
                                           min = 0, max = 1/no_cat[count] - 1e-8)
  }
  count <- 1
  for (j in new_ind){
    if (j %in% cutpoint[-1]){
      count <- count + 1
    }
    ind_active <- which(one_hot_data[, j] == 1)
    if (j %in% cutpoint){
      submatrix <- one_hot_data[, j:(j + no_cat[count] - 1)]
    }
    one_hot_data[ind_active, j] <- 2 - rowSums(submatrix)[ind_active]
  }
  return (list(one_hot_data, new_ind))
}

normalize <- function(data, numCol, parameters = NULL){
  norm_data <- data
  
  if (is.null(parameters)){
    min_val <- max_val <- rep(0, ncol(data))
    for (i in 1:ncol(data)){
      if (numCol[i] == 0){
        next
      }
      min_val[i] <- min(norm_data[, i], na.rm = T)
      norm_data[, i] <- norm_data[, i] - min(norm_data[, i], na.rm = T)
      max_val[i] <- max(norm_data[, i], na.rm = T)
      norm_data[, i] <- norm_data[, i] / (max(norm_data[, i], na.rm = T) + 1e-6)
    }
    norm_parameters <- list(min_val = min_val,
                            max_val = max_val)
  }else{
    min_val <- parameters$min_val
    max_val <- parameters$max_val
    
    for (i in 1:ncol(data)){
      norm_data[, i] <- norm_data[, i] - min_val[i]
      norm_data[, i] <- norm_data[, i] / (max_val[i] + 1e-6)
    }
    norm_parameters <- parameters
  }
  return (list(norm_data = norm_data, norm_parameters = norm_parameters))
}

renormalize <- function(norm_data, norm_parameters, numCol){
  
  min_val <- norm_parameters$min_val
  max_val <- norm_parameters$max_val
  
  renorm_data <- norm_data
  for (i in 1:ncol(norm_data)){
    if (numCol[i] == 0){
      next
    }
    renorm_data[, i] <- renorm_data[, i] * (max_val[i] + 1e-6)
    renorm_data[, i] <- renorm_data[, i] + min_val[i]
  }
  
  return (renorm_data)
  
}

torch_data <- torch::dataset(
  name = "torch_data",
  initialize = function(data){
    self$torch.data <- data
  },
  .getitem = function(ind){
    data <- self$torch.data[ind, ]
    return (list("data" = data, "index" = ind))
  },
  .length = function(){
    self$torch.data$size()[[1]]
  },
  .ncol = function(){
    self$torch.data$size()[[2]]
  }
)


new_batch <- function(norm_data, data_mask, R, misRow, batch_size, device = "cpu"){
  complete_idx <- sample(which(misRow == F))
  miss_idx <- sample(which(misRow == T))
  
  batch_complete_idx <- complete_idx[1:round(0.8 * batch_size)]
  batch_miss_idx <- miss_idx[1:(batch_size - round(0.8 * batch_size))]
  
  batch_idx <- c(batch_complete_idx, batch_miss_idx)
  
  #norm_dataloader <- torch::dataloader(norm_data, batch_size, shuffle = T)
  #norm_curr_batch <- torch::dataloader_next(torch::dataloader_make_iter(norm_dataloader))
  #mask_curr_batch <- data_mask$torch.data[norm_curr_batch$index]
  #R_curr_batch <- R[norm_curr_batch$index]
  
  norm_curr_batch <- norm_data[batch_idx, ]
  mask_curr_batch <- data_mask[batch_idx]
  R_curr_batch <- R[batch_idx]
  
  return (list(norm_curr_batch, mask_curr_batch$data$to(device = device),
               R_curr_batch$to(device = device)))
}









