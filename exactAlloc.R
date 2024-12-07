
exactAllocation <- function(data, stratum_variable, target_variable, sample_size, type){
  strata_units <- as.data.frame(table(data[[stratum_variable]]))
  colnames(strata_units) <- c(stratum_variable, "count")
  conversion_functions <- list(
    numeric = "as.numeric",
    integer = "as.integer",
    character = "as.character",
    logical = "as.logical",
    factor = "as.factor"
  )
  strata_units[, 1] <-  do.call(conversion_functions[[class(data[[stratum_variable]])[1]]], list(strata_units[, 1]))
  
  data <- merge(data, strata_units, by = stratum_variable)
  Y_bars <- aggregate(as.formula(paste0(target_variable, " ~ ", stratum_variable)), data = data, FUN = function(x) sum(x) / length(x))
  colnames(Y_bars)[2] <- "Y_bars"
  data <- merge(data, Y_bars, by = stratum_variable)
  Ss <- aggregate(as.formula(paste0("(", target_variable, " - Y_bars", ")^2", " ~ ", stratum_variable)), data = data, FUN = function(x) sum(x) / (length(x) - 1))

  NS <- strata_units$count * sqrt(Ss[, 2])
  names(NS) <- Ss[, 1]
  NS <- NS[order(NS, decreasing = T)]
  if (type == 1){
    columns <- sample_size - nrow(Ss)
    priority <- matrix(0, nrow = columns, ncol = nrow(Ss))
    colnames(priority) <- names(NS)
    for (h in names(NS)){
      priority[, h] <- NS[[h]] / sqrt((1:columns) * (2:(columns + 1)))
    }
    priority <- as.data.frame(priority)
    priority <- stack(priority)
    colnames(priority) <- c("value", stratum_variable)
    order_priority <- order(priority$value, decreasing = T)
    alloc <- (table(priority[[stratum_variable]][order_priority[1:columns]]) + 1)
    alloc <- alloc[order(as.integer(names(alloc)))]
  }else if (type == 2){
    columns <- sample_size - 2 * nrow(Ss)
    priority <- matrix(0, nrow = columns, ncol = nrow(Ss))
    colnames(priority) <- names(NS)
    for (h in names(NS)){
      priority[, h] <- NS[[h]] / sqrt((2:(columns + 1)) * (3:(columns + 2)))
    }
    priority <- as.data.frame(priority)
    priority <- stack(priority)
    colnames(priority) <- c("value", stratum_variable)
    order_priority <- order(priority$value, decreasing = T)
    alloc <- (table(priority[[stratum_variable]][order_priority[1:columns]]) + 2)
    alloc <- alloc[order(as.integer(names(alloc)))]
  }else if (type == 3){
    # 0 < ah <= nh <= bh <= Nh
    # for each column, remove all the rows <= (ah - 1)th rows, and > (bh - 1)th rows.
    columns <- sample_size
    priority <- matrix(0, nrow = columns, ncol = nrow(Ss))
    colnames(priority) <- names(NS)
    
    ah <- sapply(strata_units$count, function(c) ifelse(c <= sample_size, sample(1:(0.2 * c), 1), sample(1:(0.2 * sample_size), 1)))
    names(ah) <- strata_units[[stratum_variable]]
    bh <- mapply(function(c, a) ifelse(c <= sample_size, sample((a + 1):(0.8 * c), 1), sample((a + 1):(0.8 * sample_size), 1)), strata_units$count, ah)
    names(bh) <- strata_units[[stratum_variable]]
    for (h in names(NS)){
      priority[, h] <- (NS[[h]] / sqrt((1:columns) * (2:(columns + 1))))
      min_constraint <- priority[ah[h], h]
      max_constraint <- priority[bh[h] - 1, h]
      #priority[, h] <- ifelse(priority[, h] <= min_constraint | priority[, h] > max_constraint, 0, priority[, h])
      priority[, h] <- ifelse(priority[, h] < min_constraint, priority[, h], 0)
    }
    priority <- as.data.frame(priority)
    priority <- stack(priority)
    colnames(priority) <- c("value", stratum_variable)
    order_priority <- order(priority$value, decreasing = T)
    alloc <- (table(c(priority[[stratum_variable]][order_priority[1:(sample_size - sum(ah))]], priority[[stratum_variable]][priority$value == 0])))
    alloc <- alloc[order(as.integer(names(alloc)))]
  }else if (type == 4){
    columns <- sample_size - nrow(Ss)
    priority <- matrix(0, nrow = columns, ncol = nrow(Ss))
    colnames(priority) <- names(NS)
    for (h in names(NS)){
      priority[, h] <- NS[[h]] / sqrt((1:columns) * (2:(columns + 1)))
    }
    
  }
  return (alloc)
}
