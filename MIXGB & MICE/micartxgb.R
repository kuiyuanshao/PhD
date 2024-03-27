
micartxgb <- function(data, m = 5, iter = 5, subsample = 0.7, maxdepth = 3){
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
  cart_mixgb <- mixgb(data_for_mixgb_cart, m = m, maxit = iter, 
                      xgb.params = list(subsample = subsample, max_depth  = maxdepth))
  
  #post 
  cart_mixgb_data <- as.data.frame(bind_rows(cart_mixgb))
  resid_ind <- which(names(cart_mixgb_data) %in% paste0(colnames(cart_surrogate_avg_est), "Resid"))
  cart_mixgb_data[, col_ind] <- cart_mixgb_data[, resid_ind] + 
    do.call(rbind, replicate(m, cart_surrogate_avg_est, simplify=FALSE))
  
  return (cart_mixgb_data)
}