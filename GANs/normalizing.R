normalize.mode <- function(data, num_vars, n_modes = 2, ...) {
  if (!require(mclust, quietly = TRUE)) {
    install.packages("mclust")
    library(mclust)
  }
  data_norm <- data
  mode_params <- list()
  
  for (col in num_vars) {
    curr_col <- data[[col]]
    curr_col_obs <- curr_col[!is.na(curr_col)]
    if (length(unique(curr_col)) == 1){
      mc <- Mclust(curr_col_obs, G = 1)
    }else{
      mc <- Mclust(curr_col_obs, G = n_modes)
    }
    pred <- predict(mc, newdata = curr_col_obs)
    mode_labels <- pred$classification
    mode_means <- numeric(n_modes)
    mode_sds <- numeric(n_modes)
    
    curr_col_norm <- rep(NA, length(curr_col_obs))
    for (mode in 1:n_modes) {
      idx <- which(mode_labels == mode)
      mode_means[mode] <- mean(curr_col_obs[idx])
      mode_sds[mode] <- sd(curr_col_obs[idx]) + 1e-6
      if (is.na(sd(curr_col_obs[idx]))){
        curr_col_norm[idx] <- (curr_col_obs[idx] - mode_means[mode])
      }else{
        curr_col_norm[idx] <- (curr_col_obs[idx] - mode_means[mode]) / mode_sds[mode]
      }
    }
    mode_labels_curr_col <- rep(NA, length(curr_col))
    mode_labels_curr_col[!is.na(curr_col)] <- mode_labels
    
    curr_col[!is.na(curr_col)] <- curr_col_norm
    data_norm[[col]] <- curr_col
    data_norm[[paste0(col, "_mode")]] <- mode_labels_curr_col
    mode_params[[col]] <- list(mode_means = mode_means, mode_sds = mode_sds)
  }
  return(list(data = data_norm, mode_params = mode_params))
}

denormalize.mode <- function(data, num_vars, norm_obj){
  mode_params <- norm_obj$mode_params
  for (col in num_vars){
    curr_col <- data[[col]]
    curr_labels <- data[[paste0(col, "_mode")]]
    curr_transform <- rep(NA, length(curr_col))
    
    mode_means <- mode_params[[col]][["mode_means"]]
    mode_sds <- mode_params[[col]][["mode_sds"]]
    for (mode in unique(curr_labels)){
      idx <- which(curr_labels == mode)
      if (is.na(mode_sds[as.integer(mode)])){
        curr_transform[idx] <- curr_col[idx] + mode_means[as.integer(mode)]
      }else{
        curr_transform[idx] <- curr_col[idx] * mode_sds[as.integer(mode)] + mode_means[as.integer(mode)] 
      }
    }
    data[[col]] <- curr_transform
  }
  data_denorm <- data[, !grepl("_mode$", names(data))]
  return (list(data = data_denorm, data_mode = data))
}

normalize.minmax <- function(data, num_vars, ...){
  maxs <- apply(data, 2, max, na.rm = T)
  mins <- apply(data, 2, min, na.rm = T)
  data_norm <- do.call(cbind, lapply(names(data), function(i){
    if (i %in% num_vars){
      (data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6)
    }else{
      data[, i]
    }
  }))
  return (list(data = data_norm,
               maxs = maxs,
               mins = mins))
}

denormalize.minmax <- function(data, num_vars, norm_obj){
  maxs <- norm_obj$maxs
  mins <- norm_obj$mins
  data <- do.call(cbind, lapply(names(data), function(i){
    if (i %in% num_vars){
      data[, i] * (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6)
    }else{
      ifelse(data[, i] >= 0.5, 1, 0)
    }
  }))
  return (data)
}