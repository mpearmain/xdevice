# Simple script to merge cookies data and device data on train and validation data.
# The aim of this script is to provinde a benchmark figure as to the number of cookies a device should have
# multiple improvements can be made along the way.
#
# A simple extension is to combine top N per country number to the predicted number of cookies.

library(data.table)

# Fetch the latest nightly build using Jo-fai Chow's package
devtools::install_github("woobe/deepr")
deepr::install_h2o()
library(h2o)

# If training and validation splits of data have not been created run
# source('./cv/cv.R') to create data sets for reproducibility.

train <- fread('./data/train_basic90.csv', stringsAsFactors = T)
valid <- fread('./data/valid_basic10.csv', stringsAsFactors = T)

# Read cookie data.
cookies <- fread('./data/cookie_all_basic.csv')

# find the number of cookies per 'drawbridge-handle'
cookies.tbl <- cookies[, list('cookie_cnt' = .N), by = "drawbridge_handle"]

# We loose some unmatched values here as its a natural join.
train <- merge(train, cookies.tbl, by = 'drawbridge_handle')
valid <- merge(valid, cookies.tbl, by = 'drawbridge_handle')

# Build a quick Random Forest model to predict number of cookies.
#
# More data manips required to use other vars i.e str devtype_4 --> int(4) etc.

x0_train <- train[, c(3:11), with = FALSE]
y0_train <- train$cookie_cnt

rf0 <- randomForest(x = x0_train, y = y0_train)

# Predict on holdout
cookie_preds <- predict(rf0, valid[, c(3:11), with = FALSE])
valid[, cookie_n_pred := cookie_preds]
