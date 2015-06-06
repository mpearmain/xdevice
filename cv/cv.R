# Initial method of cutting the training set to have a validation set.
library(data.table)

full.train <- fread("./data/dev_train_basic.csv")
test <- fread("./data/dev_test_basic.csv")

# First lets get rid of drawbridge handle == -1
full.train <- full.train['drawbridge_handle' != -1, ]

# Let convert some of these strings to factors.
full.train[, device_type_f := as.factor(sub('devtype_', '', device_type))]
full.train[, device_os_f := as.factor(sub('devos_', '', device_os))]
full.train[, country_f := as.factor(sub('country_', '', country))]
full.train[, anonymous_c1_f := as.factor(sub('anonymous_c1_', '', anonymous_c1))]
full.train[, anonymous_c2_f := as.factor(sub('anonymous_c2_', '', anonymous_c2))]

full.train[, c('device_type', 'device_os', 'country', 'anonymous_c1', 'anonymous_c2') := NULL]
## repeat for test set.
test[, device_type_f := as.factor(sub('devtype_', '', device_type))]
test[, device_os_f := as.factor(sub('devos_', '', device_os))]
test[, country_f := as.factor(sub('country_', '', country))]
test[, anonymous_c1_f := as.factor(sub('anonymous_c1_', '', anonymous_c1))]
test[, anonymous_c2_f := as.factor(sub('anonymous_c2_', '', anonymous_c2))]

test[, c('device_type', 'device_os', 'country', 'anonymous_c1', 'anonymous_c2') := NULL]


# Lets cut 10% randomly
set.seed(1401081)
full.train[, rnd := runif(dim(full.train)[1])]

train <- full.train[rnd > 0.1, ]
validation <- full.train[rnd <= 0.1, ]

train[, rnd := NULL]
validation[, rnd := NULL]

write.csv(train, './data/train_basic90.csv', row.names = F)
write.csv(validation, './data/valid_basic10.csv', row.names = F)
write.csv(full.train, './data/dev_train_basic_manips.csv', row.names = F)
write.csv(test, './data/dev_test_basic_manips.csv', row.names = F)
rm('full.train', 'test', 'train', 'validation')

