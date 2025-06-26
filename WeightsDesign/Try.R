ll <- function(params, X, Y, w, alloc_p, split_q){
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

pifun <- function(Y, split, prob){
  pi <- ifelse(Y <= split[1], prob[1],
               ifelse(Y >= split[2], prob[3], prob[2]))
  return (pi)
}

load(paste0("../NutritionalData/Output/NutritionalData_", "0100", ".RData"))
data <- read.csv("Test/ODS_exactAlloc/ODS_exactAlloc_0100.csv")
split_q <- quantile(data[["c_ln_na_bio1"]], c(0.19, 0.81))
alloc_p <- table(data$R, data$outcome_strata)[2,] / colSums(table(data$R, data$outcome_strata))
ry <- data$R == 1
wy <- data$R == 0
inc <- c("c_age", "c_bmi", "c_ln_na_bio1",
         "high_chol", "usborn",
         "female", "bkg_pr", "bkg_o", "sbp", "hypertension")
x <- as.matrix(cbind(1, data[, inc]))
ynum <- data$c_ln_na_true
w <- data$W
strata <- data$outcome_strata

result <- optim(par = c(rep(0, ncol(x)), 1),
                fn = ll,
                X = x[ry, ],
                Y = ynum[ry],
                w = w[ry], 
                alloc_p = alloc_p,
                split_q = split_q,
                method = "BFGS", 
                hessian=TRUE)

coefs <- result$par[-length(result$par)]
sigma <- exp(result$par[length(result$par)])
vcov <- solve(result$hessian)
vcov <- vcov[1:ncol(x), 1:ncol(x)]

residuals <- ynum[ry] - x[ry, , drop = FALSE] %*% coefs
df <- max(length(ynum[ry]) - ncol(x[ry, , drop = FALSE]), 1)
sigma.star <- sqrt(sum(residuals^2) / rchisq(1, df))
r.c <- (t(chol(as.matrix(nearPD(vcov)$mat))) %*% rnorm(ncol(x)))
beta.star <- coefs + r.c
yhat <- numeric(length(wy))
temp_wy <- wy
while (sum(temp_wy) > 0){
  #curr_yhat <- x[temp_wy, , drop = FALSE] %*% beta.star + rnorm(sum(temp_wy)) * sigma.star
  curr_yhat <- rnorm(sum(temp_wy), mean = x[temp_wy, , drop = FALSE] %*% beta.star, sd = sigma) + rnorm(sum(temp_wy)) * sigma.star
  cond <- abs(pifun(curr_yhat, split_q, alloc_p) - 1 / w[temp_wy]) < 1e-7
  print(sum(cond))
  keep <- which(temp_wy)[cond]
  yhat[keep] <- curr_yhat[cond]
  temp_wy[keep] <- F
}
yhat <- yhat[wy]


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
    yhat
  })
}

ggplot() + 
  geom_point(aes(x = yhat,
                 y = data$sbp[data$R == 0], colour = "red")) + 
  geom_point(aes(x = pop$c_ln_na_true[data$R == 0],
                 y = data$sbp[data$R == 0]))

data$c_ln_na_true[data$R == 0] <- yhat(x[data$R == 0, ], beta.star, sigma, alloc_p, split_q)
glm(sbp ~  c_ln_na_true + 
      c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, data, family = gaussian())


pifun <- function(Y, split, prob){
  pi <- ifelse(Y <= split[1], prob[1],
               ifelse(Y >= split[2], prob[3], prob[2]))
  return (pi)
}
data <- read.csv("Test/ODS_exactAlloc/ODS_exactAlloc_0100.csv")
split_q <- quantile(data[["c_ln_na_bio1"]], c(0.19, 0.81))
alloc_p <- table(data$R, data$outcome_strata)[2,] / colSums(table(data$R, data$outcome_strata))
inc <- c("c_age", "c_bmi", "c_ln_na_bio1",
         "high_chol", "usborn",
         "female", "bkg_pr", "bkg_o", "sbp", "hypertension")
weights <- data$W
strata <- data$outcome_strata
pmm_obj <- mice(data, m = 1, print = FALSE,
                remove.collinear = F, maxit = 1,
                maxcor = 1.0001, method = "cml", ridge = 1e-5,
                predictorMatrix = quickpred(data,
                                            include = inc,
                                            exclude = names(data)[!(names(data) %in% inc)]),
                weights = weights, strata = strata, split_q = split_q, alloc_p = alloc_p,
                by = "sampling", pifun = pifun,
                pmm = F)
imputed_data_list <- lapply(1:1, function(i) complete(pmm_obj, i))
glm(hypertension ~  c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, imputed_data_list[[1]], family = binomial())
glm(sbp ~  c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, imputed_data_list[[1]], family = gaussian())



ggplot() + 
  geom_point(aes(x = imputed_data_list[[1]]$c_ln_na_true[data$R == 0],
                 y = data$sbp[data$R == 0], colour = "red")) + 
  geom_point(aes(x = data$c_ln_na_true[data$R == 1],
                 y = data$sbp[data$R == 1]))

