# Device cookie match.
# Simple script to merge device data and device data on train and validation data.
# This is the basis for a multilabel algorithm -> many:1 merge of device to cookies

library(data.table)

# If training and validation splits of data have not been created run
# source('./cv/cv.R') #to create data sets for reproducibility.
# Run feature engieering for cookies
source('./R/feat_engineer.R')
