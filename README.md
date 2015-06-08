Kaggle Xdevice competition

https://www.kaggle.com/c/icdm-2015-drawbridge-cross-device-connections

Data should be kept local, but the scripts should enable reproducible results.

## R Libs
* xgboost - devtools::install_github('dmlc/xgboost',subdir='R-package')
* data.table


## Basic Flow
1. Run the ./R/cv.R to create 90/10 % splits of the data
2. Feature engineering and formating - ./R/cookie\_n\_feat_engineering.R
3. Predicting number of cookies for a device - run ./R/cookie\_N\_pred.R

4. Feature engineering for ranking cookies to a device
5. Predicting cookie / device match
