# Initial method of cutting the training set to have a validation set.
library(data.table)

################################################################################################
# load data
{
  dev_train_basic <- fread("data/dev_train_basic.csv",stringsAsFactors =FALSE)
  dev_test_basic <- fread("data/dev_test_basic.csv",stringsAsFactors =FALSE)
  cookie_all_basic <- fread("data/cookie_all_basic.csv",stringsAsFactors =FALSE)
  #id_ip <- fread('data/id_all_ip_v2.csv', stringsAsFactors = FALSE)
}

################################################################################################
# data processing
{
#   device_ip <- id_ip[device_or_cookie_indicator == 0,]
#   device_ip$device_or_cookie_indicator <- NULL
#   colnames(device_ip)[1] <- "device_id"
#   cookie_ip <- id_ip[device_or_cookie_indicator == 1,]
#   cookie_ip$device_or_cookie_indicator <- NULL
#   colnames(cookie_ip)[1] <- "cookie_id"
#
#   # attach cookie property info to cookie_all_basic
#   cookie_all_basic <- merge(cookie_all_basic, cookie_ip, by = "cookie_id", all.x = T)
#   cookie_all_basic[is.na(cookie_all_basic)] <- -1
#
#   # attach device property to dev_train_basic
#   dev_train_basic <- merge(dev_train_basic, device_ip, by = "device_id", all.x = T)
#   dev_train_basic[is.na(dev_train_basic)] <- -1
#
#   # attach device property to dev_test_basic
#   dev_test_basic <- merge(dev_test_basic, device_ip, by = "device_id", all.x = T)
#   dev_test_basic[is.na(dev_test_basic)] <- -1


  # Let convert some of these strings to factors.
  dev_train_basic[, device_type_f := as.factor(sub('devtype_', '', device_type))]
  dev_train_basic[, device_os_f := as.factor(sub('devos_', '', device_os))]
  dev_train_basic[, country_f := as.factor(sub('country_', '', country))]
  dev_train_basic[, anonymous_c1_f := as.factor(sub('anonymous_c1_', '', anonymous_c1))]
  dev_train_basic[, anonymous_c2_f := as.factor(sub('anonymous_c2_', '', anonymous_c2))]

  dev_train_basic[, c('device_type', 'device_os', 'country', 'anonymous_c1', 'anonymous_c2') := NULL]
  ## repeat for dev_test_basic set.
  dev_test_basic[, device_type_f := as.factor(sub('devtype_', '', device_type))]
  dev_test_basic[, device_os_f := as.factor(sub('devos_', '', device_os))]
  dev_test_basic[, country_f := as.factor(sub('country_', '', country))]
  dev_test_basic[, anonymous_c1_f := as.factor(sub('anonymous_c1_', '', anonymous_c1))]
  dev_test_basic[, anonymous_c2_f := as.factor(sub('anonymous_c2_', '', anonymous_c2))]

  dev_test_basic[, c('device_type', 'device_os', 'country', 'anonymous_c1', 'anonymous_c2') := NULL]


  # Lets cut 10% randomly
  set.seed(1401081)
  dev_train_basic[, rnd := runif(dim(dev_train_basic)[1])]

  train <- dev_train_basic[rnd > 0.1, ]
  validation <- dev_train_basic[rnd <= 0.1, ]

  train[, rnd := NULL]
  validation[, rnd := NULL]

  write.csv(train, './data/train_basic90.csv', row.names = F)
  write.csv(validation, './data/valid_basic10.csv', row.names = F)
  write.csv(dev_train_basic, './data/dev_train_basic_manips.csv', row.names = F)
  write.csv(dev_test_basic, './data/dev_test_basic_manips.csv', row.names = F)
  rm('dev_train_basic', 'dev_test_basic', 'train', 'validation')
}
