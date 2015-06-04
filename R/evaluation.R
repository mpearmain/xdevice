# Simple script to calculate the F_0.5 score on data.
# (1 + \beta^2) \frac{pr}{\beta^2 p+r}\ \ 
# where: p = \frac{tp}{tp+fp}, r = \frac{tp}{tp+fn}, \beta = 0.5.

TestData <- function() {
  # Create some test data to check the F_score function.
  # For this data with Beta =1 
  # https://www.kaggle.com/wiki/MeanFScore => f_score == 0.533
  y.true <- list(c(1, 2),
                 c(3, 4, 5),
                 6,
                 7)
  y.pred <- list(c(1, 2, 3, 9),
                 c(3, 4),
                 c(6, 12),
                 1)
  data <- list("y.true" = y.true, "y.pred" = y.pred)
}

F_0.5 <- function(y.true, y.pred, beta = 0.5) {
  # y.true is a list of correct values for idx
  # y.pred is a list of predicted values for idx
  # beta the f_score value.
  
  # Check the we have identical lengths
  stopifnot(length(y.true) == length(y.pred))
  
  # Create a matrix to store tp, fp and fn
  res <- data.frame(matrix(NA, ncol = 4, nrow = length(y.true)))
  
  for(i in 1:length(y.true)) {
    res[i, 1] <-  length(intersect(y.true[[i]], y.pred[[i]])) # tp
    
    res[i, 2] <-  length(setdiff(y.pred[[i]], y.true[[i]])) # fp  
    res[i, 3] <-  length(setdiff(y.true[[i]], y.pred[[i]])) # fn
    
    r <- res[i, 1] / (res[i, 1] + res[i, 3]) # True Positive and False Negative 
    p <- res[i, 1] / (res[i, 1] + res[i, 2]) # True Positive and False Positive  
    if(res[i, 1] == 0) {
      res[i, 4] <- 0
    } else {
      res[i, 4] <- (1+ beta^2) * ((p*r) / ((beta^2 * p) + r))
    }
  }
  
  f_score <- mean(res[,4])
  return(f_score)
}