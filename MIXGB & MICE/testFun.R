testFun <- function(data, m, iter, mtry = NULL, ntree, raking = T, aux_col = 15:24, type = "MySim"){
  if (raking){
    raking_result <- cal_function(data, type = type)
  }else{
    raking_result <- NULL
  }
  #MICE with PMM
  pmm_obj <- mice(data, m = m, print = FALSE, 
                  remove.collinear = F, maxit = iter,
                  maxcor = 1.0001, 
                  predictorMatrix = quickpred(data))
  pmm_data <- complete(pmm_obj, "long")[, -c(1, 2)]
  print("PMM DONE!")
  #MICE with RF
  rf_obj <- mice(data, m = m, print = FALSE, method = "rf", 
                 remove.collinear = FALSE, 
                 maxcor = 1.0001,
                 maxit = iter, ntree = ntree, mtry = mtry)
  rf_data <- complete(rf_obj, "long")[, -c(1, 2)]
  print("RF DONE!")
  #MICE with CART
  cart_obj <- mice(data, m = m, print = FALSE, method = "cart", 
                   remove.collinear = FALSE, 
                   maxcor = 1.0001,
                   maxit = iter)
  cart_data <- complete(cart_obj, "long")[, -c(1, 2)]
  print("CART DONE!")
  #MISSFOREST
  plan(multisession, workers = 4)
  missforestimp <- function(data, maxiteriter, mtry, ntree){
    if (is.null(mtry)){
      dat = missForest(data, maxiter = iter, ntree = ntree)
    }else{
      dat = missForest(data, maxiter = iter, mtry = mtry, ntree = ntree)
    }
    return (dat$ximp)
  }
  
  mf_obj <- future_replicate(n = m, missforestimp(data, maxiter = iter, 
                                                  ntree = ntree, mtry = mtry))
  mf_obj <- lapply(seq_len(ncol(mf_obj)), function(i) mf_obj[,i])
  mf_data <- as.data.frame(bind_rows(mf_obj))
  print("MISSFOREST DONE!")
  
  #MIXGB with PMM
  mixgb_pmm <- mixgb(data, m = m, maxit = iter)
  mixgb_pmm_data <- as.data.frame(bind_rows(mixgb_pmm))
  print("MIXGBPMM DONE!")
  
  #MIXGB without PMM
  mixgb_nopmm <- mixgb(data, m = m, maxit = iter, pmm.type = NULL)
  mixgb_nopmm_data <- as.data.frame(bind_rows(mixgb_nopmm))
  print("MIXGBNOPMM DONE!")
  
  #MICE RF + MIXGB
  data_for_rf <- rbind(data, data[data$R == 1, ])
  #Creating NA entries to the new data frame, to obtain yhatobs by mice.impute.rf
  row_ind <- as.numeric(which(rowSums(is.na(data)) == 0)) #yobs row index
  col_ind <- as.numeric(which(colSums(is.na(data)) > 0)) #missing column index
  data_for_rf[(nrow(data) + 1):nrow(data_for_rf), col_ind] <- NA
  rf_surrogate <- mice(data_for_rf, m = m, print = FALSE, method = "rf", 
                       remove.collinear = FALSE, maxcor = 1.0001,
                       maxit = iter, ntree = ntree, mtry = mtry)
  rf_surrogate_data <- complete(rf_surrogate, "long")[, -c(1, 2)]
  #finding the average
  rf_surrogate_avg_est <- rowsum(rf_surrogate_data[, col_ind], 
                                 rep(1:nrow(data_for_rf), m)) / m
  #get the yobs
  yobs <- rf_surrogate_avg_est[row_ind, ]
  #get the yhatobs
  rf_yhatobs <- rf_surrogate_avg_est[(nrow(data) + 1):nrow(rf_surrogate_avg_est), ]
  #Get yhatmis and yhatobs, substitue the yhatobs to the yobs original index
  rf_surrogate_avg_est[row_ind, ] <- rf_surrogate_avg_est[(nrow(data) + 1):nrow(data_for_rf), ]
  rf_surrogate_avg_est <- as.matrix(rf_surrogate_avg_est[-((nrow(data) + 1):nrow(data_for_rf)), ])
  #X_true - X_imp
  
  rf_surrogate_resid <- matrix(NA, nrow = nrow(rf_surrogate_avg_est),
                               ncol = length(col_ind))
  rf_surrogate_resid[row_ind, ] <- as.matrix(yobs) - as.matrix(rf_yhatobs)
  data_for_mixgb_rf <- data
  #Replace X_withNA by X_imp
  data_for_mixgb_rf[, col_ind] <-  rf_surrogate_avg_est
  colnames(rf_surrogate_resid) <- paste0(colnames(rf_surrogate_avg_est), "Resid")
  data_for_mixgb_rf <- cbind(data_for_mixgb_rf, rf_surrogate_resid)
  rf_mixgb <- mixgb(data_for_mixgb_rf, m = m, maxit = iter)
  
  #post 
  rf_mixgb_data <- as.data.frame(bind_rows(rf_mixgb))
  resid_ind <- which(names(rf_mixgb_data) %in% paste0(colnames(rf_surrogate_avg_est), "Resid"))
  rf_mixgb_data[, col_ind] <- rf_mixgb_data[, resid_ind] + do.call(rbind, 
                                                                   replicate(m, rf_surrogate_avg_est, 
                                                                             simplify=FALSE))
  print("RF+MIXGB DONE!")
  #MISSFOREST + MIXGB
  data_for_mf <- rbind(data, data[data$R == 1, ])
  data_for_mf[(nrow(data) + 1):nrow(data_for_mf), col_ind] <- NA
  mf_surrogate <- future_replicate(n = m, missforestimp(data_for_mf, maxiter = iter, 
                                                        ntree = ntree, mtry = mtry))
  mf_surrogate <- lapply(seq_len(ncol(mf_surrogate)), function(i) mf_surrogate[,i])
  mf_surrogate_data <- as.data.frame(bind_rows(mf_surrogate))
  #finding the average
  mf_surrogate_avg_est <- rowsum(mf_surrogate_data[, col_ind], 
                                 rep(1:nrow(data_for_mf), m)) / m
  #get the yobs
  yobs <- mf_surrogate_avg_est[row_ind, ]
  #get the yhatobs
  mf_yhatobs <- mf_surrogate_avg_est[(nrow(data) + 1):nrow(mf_surrogate_avg_est), ]
  #Get yhatmis and yhatobs, substitue the yhatobs to the yobs original index
  mf_surrogate_avg_est[row_ind, ] <- mf_surrogate_avg_est[(nrow(data) + 1):nrow(data_for_mf), ]
  mf_surrogate_avg_est <- as.matrix(mf_surrogate_avg_est[-((nrow(data) + 1):nrow(data_for_mf)), ])
  #X_true - X_imp
  mf_surrogate_resid <- matrix(NA, nrow = nrow(mf_surrogate_avg_est),
                               ncol = length(col_ind))
  mf_surrogate_resid[row_ind, ] <- as.matrix(yobs) - as.matrix(mf_yhatobs)
  data_for_mixgb_mf <- data
  #Replace X_withNA by X_imp
  data_for_mixgb_mf[, col_ind] <-  mf_surrogate_avg_est
  colnames(mf_surrogate_resid) <- paste0(colnames(mf_surrogate_avg_est), "Resid")
  data_for_mixgb_mf <- cbind(data_for_mixgb_mf, mf_surrogate_resid)
  mf_mixgb <- mixgb(data_for_mixgb_mf, m = m, maxit = iter)
  mf_mixgb_data <- as.data.frame(bind_rows(mf_mixgb))
  
  #post 
  resid_ind <- which(names(mf_mixgb_data) %in% paste0(colnames(mf_surrogate_avg_est), "Resid"))
  mf_mixgb_data[, col_ind] <- mf_mixgb_data[, resid_ind] + do.call(rbind, 
                                                                   replicate(m, mf_surrogate_avg_est, 
                                                                             simplify=FALSE))
  print("MISSFOREST+MIXGB DONE!")
  
  data_for_cart <- rbind(data, data[data$R == 1, ])
  #Creating NA entries to the new data frame, to obtain yhatobs by mice.impute.cart
  row_ind <- as.numeric(which(rowSums(is.na(data)) == 0)) #yobs row index
  col_ind <- as.numeric(which(colSums(is.na(data)) > 0)) #missing column index
  data_for_cart[(nrow(data) + 1):nrow(data_for_cart), col_ind] <- NA
  cart_surrogate <- mice(data_for_cart, m = m, print = FALSE, method = "cart", 
                         remove.collinear = FALSE, maxcor = 1.0001,
                         maxit = iter)
  cart_surrogate_data <- complete(cart_surrogate, "long")[, -c(1, 2)]
  #finding the average
  cart_surrogate_avg_est <- rowsum(cart_surrogate_data[, col_ind], 
                                   rep(1:nrow(data_for_cart), m)) / m
  #get the yobs
  yobs <- cart_surrogate_avg_est[row_ind, ]
  #get the yhatobs
  cart_yhatobs <- cart_surrogate_avg_est[(nrow(data) + 1):nrow(cart_surrogate_avg_est), ]
  #Get yhatmis and yhatobs, substitue the yhatobs to the yobs original index
  cart_surrogate_avg_est[row_ind, ] <- cart_surrogate_avg_est[(nrow(data) + 1):nrow(data_for_cart), ]
  cart_surrogate_avg_est <- as.matrix(cart_surrogate_avg_est[-((nrow(data) + 1):nrow(data_for_cart)), ])
  
  
  #X_true - X_imp
  
  cart_surrogate_resid <- matrix(NA, nrow = nrow(cart_surrogate_avg_est),
                                 ncol = length(col_ind))
  cart_surrogate_resid[row_ind, ] <- as.matrix(yobs) - as.matrix(cart_yhatobs)
  data_for_mixgb_cart <- data
  #Replace X_withNA by X_imp
  data_for_mixgb_cart[, col_ind] <-  cart_surrogate_avg_est
  colnames(cart_surrogate_resid) <- paste0(colnames(cart_surrogate_avg_est), "Resid")
  data_for_mixgb_cart <- cbind(data_for_mixgb_cart, cart_surrogate_resid)
  cart_mixgb <- mixgb(data_for_mixgb_cart, m = m, maxit = iter)
  
  #post 
  cart_mixgb_data <- as.data.frame(bind_rows(cart_mixgb))
  resid_ind <- which(names(cart_mixgb_data) %in% paste0(colnames(cart_surrogate_avg_est), "Resid"))
  cart_mixgb_data[, col_ind] <- cart_mixgb_data[, resid_ind] + 
    do.call(rbind, replicate(m, cart_surrogate_avg_est, simplify=FALSE))
  #Naming and Joining
  
  pmm_data$mitype <- "pmm"
  rf_data$mitype <- "rf"
  cart_data$mitype <- "cart"
  mf_data$mitype <- "mf"
  mixgb_pmm_data$mitype <- "mixgb_pmm"
  mixgb_nopmm_data$mitype <- "mixgb_nopmm"
  rf_mixgb_data$mitype <- "rf+mixgb"
  mf_mixgb_data$mitype <- "mf+mixgb"
  cart_mixgb_data$mitype <- "cart+mixgb"
  
  results <- list(pmm_data, rf_data, cart_data, mf_data, mixgb_pmm_data, 
                  mixgb_nopmm_data, rf_mixgb_data, mf_mixgb_data, cart_mixgb_data,
                  raking_result)
  
  
  return (results)
}

cal_function <- function(data, type = "MySim"){
  data$id <- 1:nrow(data)
  if (type == "Simple"){
    inffun  <- dfbeta(lm(Y ~ X_tilde + Z, data=data))
    colnames(inffun) <- paste("if", 1:ncol(inffun), sep="")
    data_if <- cbind(data, inffun)
    data_if$wgt <- rep(1, nrow(data))
    
    ######
    if_design <- twophase(id = list(~id, ~id), subset = ~(R==1), weights = list(NULL, ~ wgt), 
                          data = data_if,  method='approx')
    if_cal <- calibrate(if_design, phase=2, calfun="raking", 
                        formula=~if1+if2+if3 + Z*X_tilde, data=data_if)
    res <- svyglm(Y ~ X + Z, design = if_cal)
    return (res)
  }else if (type == "MySim"){
    inffun  <- dfbeta(glm(pois_Y ~ x1star + x2star + x3star + x4star + 
                            x5star + x6star + x7star + x8star + x9star + x10star, 
                          data=data, family = poisson()))
    colnames(inffun) <- paste("if", 1:ncol(inffun), sep="")
    data_if <- cbind(data, inffun)
    data_if$wgt <- rep(1, nrow(data))
    
    ######
    if_design <- twophase(id = list(~id, ~id), subset = ~(R==1), weights = list(NULL, ~ wgt), 
                          data = data_if,  method='approx')
    if_cal <- calibrate(if_design, phase=2, calfun="raking", 
                        formula=~if1+if2+if3+if4+if5+if6+if7+if8+if9+if10+if11+
                          x1star+x2star+x3star+x4star+x5star+x6star+x7star+x8star+
                          x9star+x10star, data=data_if)
    res <- svyglm(pois_Y ~ x1 + x2 + x3 + x4 + 
                    x5 + x6 + x7 + x8 + x9 + x10, 
                  design = if_cal, family = poisson())
    return (res)
  }else if (type == "HCHS"){
    inffun1  <- dfbeta(glm(hypertension ~ c_age + c_bmi + c_ln_na_bio1 + 
                              high_chol + 
                              usborn + female + bkg_pr + bkg_o, 
                              data = data,
                              family=quasibinomial()))
    inffun2 <- dfbeta(glm(sbp ~ c_age + c_bmi + c_ln_na_bio1 + 
                             high_chol + 
                             usborn + female + bkg_pr + bkg_o,
                             data = data, 
                             family = gaussian()))
  
    colnames(inffun1) <- paste("if", 1:ncol(inffun1), sep="")
    data_if1 <- as.data.frame(cbind(data, inffun1))
    colnames(inffun2) <- paste("if", 1:ncol(inffun2), sep="")
    data_if2 <- as.data.frame(cbind(data, inffun2))
    
    data_if1$wgt <- rep(1, nrow(data))
    data_if2$wgt <- rep(1, nrow(data))
    
    if_design1 <- twophase(id = list(~id, ~id),
                           subset = ~(R == 1), 
                           weights = list(~NULL, ~wgt), 
                           data = data_if1, 
                           method='approx')
    if_design2 <- twophase(id = list(~id, ~id),
                           subset = ~(R == 1), 
                           weights = list(~NULL, ~wgt), 
                           data = data_if2, 
                           method='approx')
    
    if_cal1 <- calibrate(if_design1, phase=2, calfun="raking", 
                         formula=~if1+if2+if3+if4+
                           if5+if6+if7+if8+if9+c_ln_na_bio1, 
                         data = data_if1)
    if_cal2 <- calibrate(if_design1, phase=2, calfun="raking", 
                         formula=~if1+if2+if3+if4+
                           if5+if6+if7+if8+if9+c_ln_na_bio1, 
                         data = data_if2)
    
    res1 <- svyglm(hypertension ~ c_age + c_bmi + c_ln_na_true + 
                            high_chol + usborn + female + bkg_pr + bkg_o, 
                          family = quasibinomial(),
                          design = if_cal1)
    res2 <- svyglm(sbp ~ c_age + c_bmi + c_ln_na_true + 
                             high_chol + 
                             usborn + female + bkg_pr + bkg_o, 
                           family = gaussian(),
                           design = if_cal2)
    
    return (list(res1 = res1, res2 = res2))
  }
}

innerloop <- function(true_data, rate = 0.2, targetcols = 3:12, 
                      m = 5, iter = 5, mtry = NULL, 
                      ntree = 100, type = NULL, returndata = F, 
                      raking = T, aux_col = 18:27){
  true_data <- as.data.frame(true_data)
  data_mis <- subsel(true_data, rate = rate, 
                     targetcols = targetcols)
  col_ind <- which(colSums(is.na(data_mis)) > 0)
  results <- testFun(data_mis, m = m, iter = iter, mtry = mtry, ntree = ntree, 
                     raking, aux_col, type = type)
  new_true <- as.data.frame(true_data[, col_ind]) %>% 
    slice(rep(row_number(), m))
  RMSEtemp <- NULL
  for (k in 1:9){
    rmse <- colMeans((as.matrix(results[[k]][, col_ind]) -
                             as.matrix(new_true))^2)
    RMSEtemp <- rbind(RMSEtemp, sqrt(rmse))
  }
  if (type == "Simple"){
    m_true <- lm(Y ~ X + Z, true_data)
    true_coef <- coef(m_true)
    bias_mat_m <- matrix(NA, nrow = 9, ncol = length(coef(m_true)))
    var_mat_m_combined <- matrix(NA, nrow = 9, ncol = length(coef(m_true)))
    
    for (k in 1:9){
      curr_data_method <- results[[k]]
      coef_mat_m <- matrix(NA, nrow = m, ncol = length(coef(m_true)))
      var_mat_m <- matrix(NA, nrow = m, ncol = length(coef(m_true)))
      
      curr_m <- 1
      for (j in seq(1, nrow(new_true), by = nrow(true_data))){
        curr_fullset <- curr_data_method[j:(j + nrow(true_data) - 1), ]
        m1_imp <- lm(Y ~ X + Z, data = curr_fullset)
        coef_mat_m[curr_m, ] <- coef(m1_imp)
        
        var_mat_m[curr_m, ] <- diag(vcov(m1_imp))
        
        curr_m <- curr_m + 1
        
      }
      est_coef_m1 <- colMeans(coef_mat_m)
      
      bias_mat_m[k, ] <- est_coef_m1 - true_coef
      
      var_mat_m_combined[k, ] <- colMeans(var_mat_m) + 
        (m + 1) * diag(var(coef_mat_m)) / m
    }
    bias_mat_m <- as.data.frame(bias_mat_m)
    var_mat_m_combined <- as.data.frame(var_mat_m_combined)
    
    bias_mat_m[10, ] <- coef(results[[10]]) - true_coef
  
    bias_mat_m[, length(true_coef) + 1] <- c("pmm", "rf", "cart", "mf", 
                           "mixgb_pmm", "mixgb_nopmm",
                           "rf+mixgb", "mf+mixgb", "cart+mixgb", 
                           "raking")
    
    var_mat_m_combined[10, ] <- diag(vcov(m_true))
    
    var_mat_m_combined[11, ] <- diag(vcov(results[[10]]))
    var_mat_m_combined[, length(true_coef) + 1] <- c("pmm", "rf", "cart", "mf", 
                                   "mixgb_pmm", "mixgb_nopmm",
                                   "rf+mixgb", "mf+mixgb", 
                                   "cart+mixgb", 
                                   "true",
                                   "raking")
    
    names(bias_mat_m) <- c(names(coef(m_true)), "mitype")
    
    names(var_mat_m_combined) <- c(names(coef(m_true)), "mitype")
    return (list(RMSEtemp, bias_mat_m, var_mat_m_combined))
  }
  
  if (type == "HCHS"){
    m1_true <- glm(hypertension ~ c_age + c_bmi + c_ln_na_true + 
                        high_chol + usborn + female + bkg_pr + bkg_o, 
                      data = true_data, 
                      family = quasibinomial())
    m2_true <- glm(sbp ~ c_age + c_bmi + c_ln_na_true + high_chol + 
                        usborn + female + bkg_pr + bkg_o,
                      data = true_data, 
                      family = gaussian())
    true_coef <- list(coef(m1_true), coef(m2_true))
    true_var <- list(diag(vcov(m1_true)), diag(vcov(m2_true)))
    
    bias_mat_m1 <- matrix(NA, nrow = 9, ncol = length(coef(m1_true)))
    bias_mat_m2 <- matrix(NA, nrow = 9, ncol = length(coef(m2_true)))
    
    var_mat_m1_combined <- matrix(NA, nrow = 9, ncol = length(coef(m1_true)))
    var_mat_m2_combined <- matrix(NA, nrow = 9, ncol = length(coef(m2_true)))
    for (k in 1:9){
      curr_data_method <- results[[k]]
      coef_mat_m1 <- matrix(NA, nrow = m, ncol = length(coef(m1_true)))
      coef_mat_m2 <- matrix(NA, nrow = m, ncol = length(coef(m2_true)))
      var_mat_m1 <- matrix(NA, nrow = m, ncol = length(coef(m1_true)))
      var_mat_m2 <- matrix(NA, nrow = m, ncol = length(coef(m1_true)))
      
      curr_m <- 1
      for (j in seq(1, nrow(new_true), by = nrow(true_data))){
        curr_fullset <- curr_data_method[j:(j + nrow(true_data) - 1), ]

        m1_imp <- glm(hypertension ~ c_age + c_bmi + c_ln_na_true + 
                            high_chol + usborn + female + bkg_pr + bkg_o, 
                         data = curr_fullset, 
                          family = quasibinomial())
        m2_imp <- glm(sbp ~ c_age + c_bmi + c_ln_na_true + 
                           high_chol + usborn + female + bkg_pr + bkg_o,
                         data = curr_fullset, 
                         family = gaussian())
        coef_mat_m1[curr_m, ] <- coef(m1_imp)
        coef_mat_m2[curr_m, ] <- coef(m2_imp)
        
        var_mat_m1[curr_m, ] <- diag(vcov(m1_imp))
        var_mat_m2[curr_m, ] <- diag(vcov(m2_imp))
        
        curr_m <- curr_m + 1
        
      }
      est_coef_m1 <- colMeans(coef_mat_m1)
      est_coef_m2 <- colMeans(coef_mat_m2)
      
      bias_mat_m1[k, ] <- est_coef_m1 - true_coef[[1]]
      bias_mat_m2[k, ] <- est_coef_m2 - true_coef[[2]]
      
      var_mat_m1_combined[k, ] <- colMeans(var_mat_m1) + 
        (m + 1) * diag(var(coef_mat_m1)) / m
      var_mat_m2_combined[k, ] <- colMeans(var_mat_m2) + 
        (m + 1) * diag(var(coef_mat_m2)) / m
    }
    bias_mat_m1 <- as.data.frame(bias_mat_m1)
    bias_mat_m2 <- as.data.frame(bias_mat_m2)
    var_mat_m1_combined <- as.data.frame(var_mat_m1_combined)
    var_mat_m2_combined <- as.data.frame(var_mat_m2_combined)
    
    bias_mat_m1[10, ] <- coef(results[[10]][[1]]) - true_coef[[1]]
    bias_mat_m2[10, ] <- coef(results[[10]][[2]]) - true_coef[[2]]
    
    bias_mat_m1[, 10] <- c("pmm", "rf", "cart", "mf", 
                           "mixgb_pmm", "mixgb_nopmm",
                           "rf+mixgb", "mf+mixgb", "cart+mixgb", 
                           "raking")
    bias_mat_m2[, 10] <- c("pmm", "rf", "cart", "mf", 
                           "mixgb_pmm", "mixgb_nopmm",
                           "rf+mixgb", "mf+mixgb", "cart+mixgb", 
                           "raking")
    
    var_mat_m1_combined[10, ] <- true_var[[1]]
    var_mat_m2_combined[10, ] <- true_var[[2]]
    var_mat_m1_combined[11, ] <- diag(vcov(results[[10]][[1]]))
    var_mat_m2_combined[11, ] <- diag(vcov(results[[10]][[2]]))
    
    var_mat_m1_combined[, 10] <- c("pmm", "rf", "cart", "mf", 
                                   "mixgb_pmm", "mixgb_nopmm",
                                   "rf+mixgb", "mf+mixgb", 
                                   "cart+mixgb", 
                                   "true", 
                                   "raking")
    var_mat_m2_combined[, 10] <- c("pmm", "rf", "cart", "mf", 
                                   "mixgb_pmm", "mixgb_nopmm",
                                   "rf+mixgb", "mf+mixgb", 
                                   "cart+mixgb", 
                                   "true",
                                   "raking")
    
    names(bias_mat_m1) <- c(names(coef(m1_true)), "mitype")
    names(bias_mat_m2) <- c(names(coef(m2_true)), "mitype")
    
    names(var_mat_m1_combined) <- c(names(coef(m2_true)), "mitype")
    names(var_mat_m2_combined) <- c(names(coef(m2_true)), "mitype")
    return (list(RMSEtemp, bias_mat_m1, bias_mat_m2, 
                 var_mat_m1_combined, var_mat_m2_combined))
  }
  
  return (RMSEtemp)
}
