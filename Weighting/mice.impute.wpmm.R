mice.impute.weightedpmm <- function(y, ry, x, wy = NULL, donors = 5L, matchtype = 1L, 
                                    ridge = 1e-05, use.matcher = FALSE, w, ...) {
  
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
  xtwy <- t(x[ry, , drop = FALSE]) %*% (ynum[ry] * w[ry])
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
  
  r.c <- (t(chol(sym(v))) %*% rnorm(ncol(x))) * sigma.star
  r.c <- r.c[order(lm.fit(x = x[ry, , drop = FALSE], y = y[ry])$qr$pivot), ]
  beta.star <- c + r.c
  
  if (matchtype == 0L) {
    yhatobs <- x[ry, , drop = FALSE] %*% c
    yhatmis <- x[wy, , drop = FALSE] %*% c
  }
  if (matchtype == 1L) {
    yhatobs <- x[ry, , drop = FALSE] %*% c
    yhatmis <- x[wy, , drop = FALSE] %*% beta.star
  }
  if (matchtype == 2L) {
    yhatobs <- x[ry, , drop = FALSE] %*% beta.star
    yhatmis <- x[wy, , drop = FALSE] %*% beta.star
  }
  if (use.matcher) {
    idx <- matcher(yhatobs, yhatmis, k = donors)
  }
  else {
    idx <- matchindex(yhatobs, yhatmis, donors)
  }
  
  return(y[ry][idx])
}