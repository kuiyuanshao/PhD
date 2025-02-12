mice.impute.snorm <- function(y, ry, x, wy = NULL, 
                                ridge = 1e-05, use.matcher = FALSE, ...) {
  args <- list(...)
  strata <- args$strata
  
  
  {
    if (is.null(wy)) {
      wy <- !ry
    }
  }
  x <- cbind(1, as.matrix(x))
  ynum <- y
  if (is.factor(y)) {
    ynum <- as.integer(y)
  }
  
  imputed <- rep(NA, length(ynum))
  
  unique_strata <- unique(strata)
  
  # Loop over each stratum
  for (s in unique_strata) {
    idx <- which(strata == s)
    curr_y <- ynum[idx]
    curr_x <- x[idx, , drop = FALSE]
    curr_ry <- ry[idx]
    curr_wy <- wy[idx]
    
    if (sum(curr_ry) == 0) next
    if (sum(curr_wy) < 1) next
    
    x_obs <- curr_x[curr_ry, , drop = FALSE]
    y_obs <- curr_y[curr_ry]
    
    xtx <- t(x_obs) %*% x_obs
    xty <- t(y_obs) %*% x_obs
    
    pen <- ridge * diag(xtx)
    if (length(pen) == 1) {
      pen <- matrix(pen)
    }
    
    v <- solve(xtx + diag(pen))
    c_coef <- xty %*% v
    
    r_resid <- y_obs - x_obs %*% t(c_coef)
    df <- max(length(y_obs) - ncol(x_obs), 1)
    sigma.star <- sqrt(sum(r_resid^2) / rchisq(1, df))
    
    if (any(is.na(c_coef))) {
      c_coef[is.na(c_coef)] <- 0
    }
    v <- (v + t(v)) / 2
    
    r_c <- as.vector(t(chol(v)) %*% rnorm(ncol(x_obs))) * sigma.star
    
    fit <- lm.fit(x = x_obs, y = y_obs)
    r_c <- r_c[order(fit$qr$pivot)]
    
    beta.star <- as.vector(t(c_coef)) + as.vector(r_c)
    
    x_missing <- curr_x[curr_wy, , drop = FALSE]
    imputed_values <- as.vector(x_missing %*% beta.star + rnorm(sum(curr_wy)) * sigma.star)
    imputed[idx[curr_wy]] <- imputed_values
  }
  
  imputed[wy]

}
