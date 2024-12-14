mice.impute.pmm <- function (y, ry, x, wy = NULL, donors = 5L, matchtype = 1L, 
          ridge = 1e-05, use.matcher = FALSE, ...) 
{
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
  parm <- .norm.draw(ynum, ry, x, ridge = ridge, ...)
  if (matchtype == 0L) {
    yhatobs <- x[ry, , drop = FALSE] %*% parm$coef
    yhatmis <- x[wy, , drop = FALSE] %*% parm$coef
  }
  if (matchtype == 1L) {
    yhatobs <- x[ry, , drop = FALSE] %*% parm$coef
    yhatmis <- x[wy, , drop = FALSE] %*% parm$beta
  }
  if (matchtype == 2L) {
    yhatobs <- x[ry, , drop = FALSE] %*% parm$beta
    yhatmis <- x[wy, , drop = FALSE] %*% parm$beta
  }
  if (use.matcher) {
    idx <- matcher(yhatobs, yhatmis, k = donors)
  }
  else {
    idx <- matchindex(yhatobs, yhatmis, donors)
  }
  
  return(y[ry][idx])
}

assignInNamespace("mice.impute.pmm", mice.impute.pmm, ns = "mice")



retrieve <- function(data, var, predM, ry, wy, ls.meth, ridge = 1e-5){
  ynum <- data[[var]]
  x <- as.matrix(data[, predM[var, ] == 1])
  df <- max(length(ynum[ry]) - ncol(x[ry, ]), 1)
  if (ls.meth == "qr"){
    qr <- lm.fit(x = x[ry, ], y = ynum[ry])
    c <- qr$coef
    f <- qr$fitted.values
    r <- t(qr$residuals)
    xtx <- as.matrix(crossprod(qr.R(qr$qr)))
    pen <- diag(xtx) * ridge
    v <- solve(xtx + diag(pen))
    v <- (v + t(v)) / 2
    #finding the residual variance via a random drawing from a posterior distribution
    sigma.star <- sqrt(sum((r)^2) / rchisq(1, df))
    random.component <- (chol(v) %*% rnorm(ncol(x))) * sigma.star
    beta.star <- c + random.component[order(qr$qr$pivot), ]
    
    c[is.na(c)] <- 0
    beta.star[is.na(beta.star)] <- 0
    
    yhatobs <- x[ry, , drop = FALSE] %*% c
    yhatmis <- x[wy, , drop = FALSE] %*% beta.star
    ypmmmis <- pmm(yhatobs, yhatmis, ynum[ry], 5)
    result <- list(coef = c, beta = beta.star, resvar = sigma.star, xtxinv = v, 
                   yhatobs = yhatobs, yhatmis = yhatmis, ypmmmis = ypmmmis,
                   qr = qr)
    return (result)
  }else if (ls.meth == "ridge"){
    xtx <- crossprod(x)
    pen <- ridge * diag(xtx)
    v <- solve(xtx + diag(pen))
    c <- t(y) %*% x %*% v
    r <- ynum[ry] - x[ry, ] %*% t(c)
    sigma.star <- sqrt(sum((r)^2) / rchisq(1, df))
    v <- (v + t(v)) / 2
    beta.star <- c + (t(chol(v)) %*% rnorm(ncol(x))) * sigma.star
    
    c[is.na(c)] <- 0
    beta[is.na(beta)] <- 0
    
    yhatobs <- x[ry, , drop = FALSE] %*% c
    yhatmis <- x[wy, , drop = FALSE] %*% beta
    ypmmmis <- pmm(yhatobs, yhatmis, ynum[ry], 5)
    result <- list(coef = c, beta = beta.star, resvar = sigma.star, xtxinv = v, 
                   yhatobs = yhatobs, yhatmis = yhatmis, ypmmmis = ypmmmis)
    return (result)
  }
}
