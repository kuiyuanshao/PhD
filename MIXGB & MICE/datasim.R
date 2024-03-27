pacman::p_load(mvtnorm)

simtwophasepaper <- function(SD = c(sqrt(3), sqrt(3))){
  set.seed(210818)
  
  Nsim <- 3000
  n  <- 10000
  n2 <- 2000
  ######################################
  
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

#simulation of multinomial distribution of sex and category, sex * category
sim_sexcat <- function(data, prob, sex = c("M", "F"), category = c("A", "B", "C", "D")){
  tb <- expand.grid(sex,category)
  colnames(tb) <- c('sex', 'category')
  tb$index <- 1:nrow(tb)
  tb$prob <- prob
  
  data$index <- apply(rmultinom(nrow(data), size = 1, prob = tb$prob), 2, which.max)
  data <- merge(data, subset(tb, select=-prob), by='index', all.x = T)
  
  return (data)
}

sim_covariates <- function(data, mu, sigma, ncov = 10, sd){
  covariates <- paste0("x", 1:ncov)
  indices <- unique(data$index)
  df <- NULL
  for (i in indices){
    sub <- subset(data, index == i)
    curr_m <- mu[i, ]
    curr_v <- matrix(sigma[i, ], ncov, ncov, byrow = T)
    curr_v[lower.tri(curr_v)] <- t(curr_v)[lower.tri(curr_v)]
    
    sub <- cbind(sub, rmvnorm(nrow(sub), mean = curr_m, sigma = curr_v, method = "svd"))
    colnames(sub) <- c(colnames(data), covariates)
    df <- rbind(df, sub)
  }
  
  
  lm_Y <- 10
  pois_Y <- 0
  for (j in 1:ncov){
    df[, j + 3] <- scales::rescale(df[, j + 3], to = 0.1 * i * c(1, 10))
    lm_Y <- lm_Y + sample(seq(-2, 2), 1) * df[, j + 3]
    pois_Y <- pois_Y + sample(seq(-2, 2, by = 1), 1) * df[, j + 3]
  }

  df$lm_Y <- lm_Y + rnorm(nrow(df))
  df$pois_Y <- rpois(5000, lambda = exp(pois_Y))
  #Add measurement error
  for (i in 1:ncov){
    df <- cbind(df, df[, i + 3] + rnorm(nrow(df), 0, sd))
  }
  #rnorm(nrow(df), 0, sd(df[, i + 3]) / 2))
  #sd(df[, i + 3]) / 2)
  colnames(df) <- c(colnames(df)[1:(5 + ncov)], paste0("x", 1:ncov, "star"))
  
  return (df)
}

sim_mean <- function(ncov){
  mu <- NULL
  for (i in 1:ncov){
    current <- rnorm(8, sample(1:100, 1))
    mu <- cbind(mu, current)
  }
  return (mu)
}

sim_variance <- function(ncov){
  sigma <- NULL
  for (i in 1:ncov^2){
    current <- rnorm(8, sample(1:10, 1), sample(1:5, 1))
    sigma <- cbind(current, sigma)
  }
  return (abs(sigma))
}

subsel <- function(data, rate = 0.2, targetcols = c(3:12)){
  n <- nrow(data)
  n2 <- as.integer(n * rate)
  id_phase2 <- c(sample(n, n2))
  for (i in targetcols){
    data[-id_phase2, i] <- NA 
  }
  data$R <- 0
  data$R[id_phase2] <- 1
  return (data)
}

dataSim <- function(mu, sigma, ncov, sd){
  
  data <- as.data.frame(1:5000)
  data <- sim_sexcat(data, prob)
  data <- data[, -2]
  data <- sim_covariates(data, mu, sigma, ncov = 10, sd = sd)
  data <- data[, -1]
  
  return (data)
}

