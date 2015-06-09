# Simple script to merge cookies data and device data on train and validation data.
# The aim of this script is to provinde a benchmark figure as to the number of cookies a device should have
# multiple improvements can be made along the way.
#
# A simple extension is to combine top N per country number to the predicted number of cookies.

library(data.table)
require(xgboost)
require(methods)

# If training and validation splits of data have not been created run
source('./cv/cv.R') #to create data sets for reproducibility.
# Run feature engieering for cookies
source('./R/cookie_n_feat_engineer.R')


# Read cookie data.
cookies <- fread('./data/cookie_all_basic.csv')
# find the number of cookies per 'drawbridge-handle'
cookies.tbl <- cookies[, list('cookie_cnt' = .N), by = "drawbridge_handle"]
# We loose some unmatched values here as its a natural join.
train <- merge(train, cookies.tbl, by = 'drawbridge_handle')
valid <- merge(valid, cookies.tbl, by = 'drawbridge_handle')
# Its a big flaw doing this as classification as it bounds the options,
# however i think it'll get us a long way just predicting more than #1 cookie slot.
train[, cookie_cnt := as.factor(cookie_cnt)]
valid[, cookie_cnt := as.factor(cookie_cnt)]

# Build a quick xgboost model to predict number of cookies.
x0_train <- train[, c(3:11), with = FALSE]
x0_valid <- valid[, c(3:11), with = FALSE]
x0_full_train <- full.train[, c(3:11), with = FALSE]
x0_test <- test[, c(3:11), with = FALSE]

y <- train[, cookie_cnt]
y <- as.integer(y)-1 #xgboost take features in [0,numOfClass)

x <- rbind(x0_train, x0_valid)
x <- as.matrix(x)
x <- matrix(as.numeric(x), nrow(x), ncol(x))
trind <- 1:length(y)
teind <- (nrow(train)+1):nrow(x)

# Set necessary parameter - should change to softprob for ensembles later
# This is just easy now.
param <- list("objective" = "multi:softmax",
              "eval_metric" = "mlogloss",
              "num_class" = length(table(y)),
              "nthread" = 8,
              "gamma" = 0.1,
              "colsample_bytree" = 0.91,
              "min_child_weight" = 3,
              "max_depth" = 15)

# # Run Cross Valication
# cv.nround <- 250
# bst.cv <- xgb.cv(param=param, 
#                 data = x[trind,], 
#                 label = y, 
#                 nfold = 5, 
#                 nrounds=cv.nround,
#                 "eta"=0.1)

# Train the model - AWFUL RESULTS ATM
nround <- 500
bst <- xgboost(param=param, 
              data = x[trind,], 
              label = y,
              nrounds=nround, 
              "eta"=0.1,
              early.stop.round = 5)

# Make prediction - remember to add 1 to re-align
pred <- predict(bst,x[teind,]) + 1
valid <- cbind(valid, pred)
