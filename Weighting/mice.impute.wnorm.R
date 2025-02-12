mice.impute.wnorm <- function(y, ry, x, wy = NULL, 
                                     ridge = 1e-05, use.matcher = FALSE, ...) {
  args <- list(...)
  w <- args$weights
  
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
  c <- xtwy %*% v
  r <- ynum[ry] - x[ry, , drop = FALSE] %*% t(c)
  
  df <- max(length(ynum[ry]) - ncol(x[ry, , drop = FALSE]), 1)
  sigma.star <- sqrt(sum(r^2) / rchisq(1, df))
  if (any(is.na(c))) {
    c[is.na(c)] <- 0
  }
  
  v <- (v + t(v)) / 2
  
  r.c <- (t(chol(v)) %*% rnorm(ncol(x))) * sigma.star
  r.c <- r.c[order(lm.fit(x = x[ry, , drop = FALSE], y = y[ry])$qr$pivot), ]
  
  c <- t(c)
  beta.star <- c + r.c
  x[wy, ] %*% beta.star + rnorm(sum(wy)) * sigma.star
}
