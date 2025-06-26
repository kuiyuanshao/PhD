library(Matrix)
mice.impute.wnorm <- function(y, ry, x, wy = NULL, 
                              ridge = 1e-05, use.matcher = FALSE, ...) {
  args <- list(...)
  w <- args$weights
  pmm <- args$pmm
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
  
  xtwx <- t(x[ry, , drop = FALSE]) %*% (x[ry, , drop = FALSE] * w[ry])
  xtwy <- t(ynum[ry] * w[ry]) %*% (x[ry, , drop = FALSE])
  pen <- ridge * diag(xtwx)
  if (length(pen) == 1) {
    pen <- matrix(pen)
  }
  v <- solve(xtwx + diag(pen))
  coefs <- xtwy %*% v
  residuals <- ynum[ry] - x[ry, , drop = FALSE] %*% t(coefs)

  v_sandwich <- t(v) %*% (t(x[ry, , drop = FALSE]) %*% diag(w[ry]) %*% 
                            diag(as.vector(residuals^2)) %*% diag(w[ry]) %*% x[ry, , drop = FALSE]) %*% v
  
  df <- max(length(ynum[ry]) - ncol(x[ry, , drop = FALSE]) - unique(w), 1)
  sigma.star <- sqrt(sum(residuals^2) / rchisq(1, df))
  
  if (any(is.na(coefs))) {
    coefs[is.na(coefs)] <- 0
  }
  
  v_sandwich <- (v_sandwich + t(v_sandwich)) / 2  # Ensure symmetry
  r.c <- (t(chol(as.matrix(nearPD(v_sandwich)$mat))) %*% rnorm(ncol(x)))
  r.c <- r.c[order(lm.fit(x = x[ry, , drop = FALSE], y = y[ry])$qr$pivot), ]
  
  beta.star <- t(coefs) + r.c
  if (pmm){
    yhatobs <- x[ry, , drop = FALSE] %*% coefs
    yhatmis <- x[wy, , drop = FALSE] %*% beta.star
    idx <- matchindex(yhatobs, yhatmis, 5)
    return (y[ry][idx])
  }else{
    return (x[wy, ] %*% beta.star + rnorm(sum(wy)) * sigma.star)
  }
}
