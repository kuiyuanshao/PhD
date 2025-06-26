generateLinearData <- function(digit){
    beta <- c(1, 1, 1)
    e_U <- c(sqrt(3), sqrt(3))
    mx <- 0; sx <- 1; zrange <- 1; zprob <- .5
    simZ   <- rbinom(10000, zrange, zprob) + 1
    simX   <- (1-simZ)*rnorm(10000, 0, 1) + simZ*rnorm(10000, 0.5, 1)
    epsilon <- rnorm(10000, 0, 1)
    simY    <- beta[1] + beta[2]*simX + beta[3]*simZ + epsilon
    simX_tilde <- simX + rnorm(10000, 0, e_U[1]*(simZ==1) + e_U[2]*(simZ==2))
    data <- data.frame(X_tilde=simX_tilde, Y=simY, X=simX, Z=simZ)
    if(!dir.exists('/nesi/project/uoa03789/PhD/SamplingDesigns/SimpleLinearData/Output')){system('mkdir /nesi/project/uoa03789/PhD/SamplingDesigns/SimpleLinearData/Output')}
    save(data,file=paste0('/nesi/project/uoa03789/PhD/SamplingDesigns/SimpleLinearData/Output/SimpleLinearData_', digit, '.RData'),compress = 'xz')
}