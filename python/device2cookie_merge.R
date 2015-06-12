# Device cookie match.
# Simple script to merge device data and device data on train and validation data.
# This is the basis for a multilabel algorithm -> many:1 merge of device to cookies

# The idea is to use the device and cookie features to predict which cookies are 'most likely' for a device.
# Combining this with the N number of cookies we expect for each device we have a framework to improve upon.

library(data.table)

# If training and validation splits of data have not been created run
# source('./cv/cv.R') #to create data sets for reproducibility.


# Read cookie data.
cookies <- fread('./data/cookie_all_basic.csv')

# Run feature engieering for device
source('./R/feat_engineer.R')
# Generates train and valid
train.c <- merge(train, cookies, by = 'drawbridge_handle')
valid.c <- merge(valid, cookies, by = 'drawbridge_handle')

