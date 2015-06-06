# Simple script to merge cookies data and device data on train and validation data.
# The aim of this script is to provinde a benchmark figure as to the number of cookies a device should have
# multiple improvements can be made along the way.
#
# A simple extension is to combine top N per country number to the predicted number of cookies.

library(data.table)

# Fetch the latest nightly build using Jo-fai Chow's package
# devtools::install_github("woobe/deepr")
deepr::install_h2o()
library(h2o)

# If training and validation splits of data have not been created run
source('./cv/cv.R') #to create data sets for reproducibility.

train <- fread('./data/train_basic90.csv', stringsAsFactors = T)
valid <- fread('./data/valid_basic10.csv', stringsAsFactors = T)
full.train <- fread('./data/dev_train_basic_manips.csv', stringsAsFactors = T)
test <- fread('./data/dev_test_basic_manips.csv', stringsAsFactors = T)

train[, device_type_f := as.factor(device_type_f)]
train[, device_os_f:= as.factor(device_os_f)]
train[, country_f := as.factor(country_f)]
train[, anonymous_c1_f := as.factor(anonymous_c1_f)]
train[, anonymous_c2_f := as.factor(anonymous_c2_f)]

valid[, device_type_f := as.factor(device_type_f)]
valid[, device_os_f:= as.factor(device_os_f)]
valid[, country_f := as.factor(country_f)]
valid[, anonymous_c1_f := as.factor(anonymous_c1_f)]
valid[, anonymous_c2_f := as.factor(anonymous_c2_f)]

full.train[, device_type_f := as.factor(device_type_f)]
full.train[, device_os_f:= as.factor(device_os_f)]
full.train[, country_f := as.factor(country_f)]
full.train[, anonymous_c1_f := as.factor(anonymous_c1_f)]
full.train[, anonymous_c2_f := as.factor(anonymous_c2_f)]

test[, device_type_f := as.factor(device_type_f)]
test[, device_os_f:= as.factor(device_os_f)]
test[, country_f := as.factor(country_f)]
test[, anonymous_c1_f := as.factor(anonymous_c1_f)]
test[, anonymous_c2_f := as.factor(anonymous_c2_f)]


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

# Build a quick Random Forest model to predict number of cookies.
#
# randomForest here is painful as it takes a Looooooong time, hence h20 its also easy to switch to
# gbm if required.

localH2O = h2o.init(max_mem_size = "20g", nthreads = 8)

x0_train <- as.h2o(train[, c(3:12), with = FALSE])
x0_valid <- as.h2o(valid[, c(3:12), with = FALSE])
x0_full_train <- as.h2o(full.train[, c(3:12), with = FALSE])
x0_test <- as.h2o(test[, c(3:12), with = FALSE])


gbm.h20 <- h2o.gbm(x = names(x0_train)[1:ncol(x0_train) -1],
                  learn_rate = 0.3,
                  y = names(x0_train)[ncol(x0_train)],
                  training_frame = x0_full_train,
                  ntrees = 1000)
y_pred <- as.data.frame(h2o.predict(gbm.h20, x0_test))
test[, preds := y_pred$predict]

write.csv(test, './data/dev_test_basic_N_cookies.csv', row.names = F)




