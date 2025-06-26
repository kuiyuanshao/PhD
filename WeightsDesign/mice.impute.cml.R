library(Matrix)
mice.impute.cml <- function(y, ry, x, wy = NULL, 
                              ridge = 1e-05, use.matcher = FALSE, ...) {
  args <- list(...)
  w <- args$weights
  split_q <- args$split_q
  alloc_p <- args$alloc_p
  by <- args$by
  pifun <- args$pifun
  
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
  
  ll <- function(params, X, Y, alloc_p, split_q){
    betas <- params[-length(params)]
    sigma <- exp(params[length(params)]) #result of sigma will be log(sigma)
  
    mu <- as.vector(X %*% betas)

    log_f <- dnorm(Y, mean = mu, sd = sigma, log = TRUE)
    log_pi <- log(1 / w)
    pnorm_vals <- pnorm(split_q, mean = mu, sd = sigma)
    integral <- alloc_p[1] * pnorm_vals[1] +
      (if(length(alloc_p) > 2) sum(alloc_p[2:(length(alloc_p)-1)] * diff(pnorm_vals)) else 0) +
      alloc_p[length(alloc_p)] * (1 - pnorm_vals[length(pnorm_vals)])
    log_den <- log(integral)
    
    loglik <- log_f + log_pi - log_den
    
    return(-sum(loglik))
  }
  
  result <- optim(par = c(rep(0, ncol(x)), 1),
                  fn = ll,
                  X = x[ry, ],
                  Y = ynum[ry],
                  alloc_p = alloc_p,
                  split_q = split_q,
                  method = "BFGS", 
                  hessian=TRUE)
  
  coefs <- result$par[-length(result$par)]
  sigma <- exp(result$par[length(result$par)])
  
  pen <- ridge * diag(result$hessian)
  if (length(pen) == 1) {
    pen <- matrix(pen)
  }
  
  vcov <- solve(result$hessian + diag(pen))
  vcov <- vcov[1:ncol(x), 1:ncol(x)]
  
  residuals <- ynum[ry] - x[ry, , drop = FALSE] %*% coefs
  df <- max(length(ynum[ry]) - ncol(x[ry, , drop = FALSE]), 1)
  sigma.star <- sqrt(sum(residuals^2) / rchisq(1, df))

  r.c <- (t(chol(as.matrix(nearPD(vcov)$mat))) %*% rnorm(ncol(x)))
  beta.star <- coefs + r.c
  
  if (by == "integration"){
    yhat <- function(x, betas, sigma, alloc_p, split_q) {
      apply(x, 1, function(x) {
        alloc_p <- 1 - alloc_p
        mu <- sum(x * betas)
        pnorm_vals <- pnorm(split_q, mean = mu, sd = sigma)
        Dx <- alloc_p[1] * pnorm_vals[1] +
          (if(length(alloc_p) > 2) sum(alloc_p[2:(length(alloc_p)-1)] * diff(pnorm_vals)) else 0) +
          alloc_p[length(alloc_p)] * (1 - pnorm_vals[length(pnorm_vals)])
        NxFun <- function(y) {
          y * dnorm(y, mean = sum(x * betas), sd = sigma)
        }
        Nx_1 <- integrate(NxFun, lower = -Inf, upper = split_q[1])$value
        Nx_2 <- integrate(NxFun, lower = split_q[1], upper = split_q[2])$value
        Nx_3 <- integrate(NxFun, lower = split_q[2], upper = Inf)$value
        Nx <- alloc_p[1] * Nx_1 + alloc_p[2] * Nx_2 + alloc_p[3] * Nx_3
        yhat <- Nx / Dx
      })
    }
    yhat <- yhat(x[wy, ], beta.star, sigma, alloc_p, split_q) + rnorm(sum(wy)) * sigma.star
  }else if (by == "sampling"){
    yhatmat <- NULL
    for (i in 1:1){
      yhat <- numeric(length(wy))
      temp_wy <- wy
      while (sum(temp_wy) > 0){
        #curr_yhat <- x[temp_wy, , drop = FALSE] %*% beta.star + rnorm(sum(temp_wy)) * sigma.star
        curr_yhat <- rnorm(sum(temp_wy), mean = x[temp_wy, , drop = FALSE] %*% beta.star, sd = sigma) #+ rnorm(sum(temp_wy)) * sigma.star
        p_acc <- (1 - pifun(curr_yhat, split_q, alloc_p)) / (1 - max(alloc_p))
        u <- runif(sum(temp_wy))
        cond <- u < p_acc
        #cond <- abs(pifun(curr_yhat, split_q, alloc_p) - 1 / w[temp_wy]) < 1e-7
        keep <- which(temp_wy)[cond]
        yhat[keep] <- curr_yhat[cond]
        temp_wy[keep] <- F
      }
      yhat <- yhat[wy] #+ rnorm(sum(wy)) * sigma.star
      yhatmat <- cbind(yhatmat, yhat)
    }
    yhat <- rowMeans(yhatmat)
    #yhat <- apply(yhatmat, 1, median, na.rm = T)
  }
  return (yhat)
}