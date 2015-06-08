# Feature engineering.
# Pre process ffeatures and factors to use in all subsequent cookie_N_prediction.

library(data.table)
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