################################################################################################
# wd etc
{
  library(data.table)
  library(plyr)
  projPath <- "/Users/konrad/Documents/projects/xdevice"
  setwd(projPath)
}

################################################################################################
# extra functions
{
  myFun <- function(x)
  {
    tbl <- table(x$cookie_id)
    x$cookie_id <- rep(names(tbl)[which.max(tbl)],nrow(x))
    x
  }
  
  # F 0.5 score
  f05score <- function(true, pred)
  {
    tp <- sum( (true == 1) & (pred == 1))
    fp <- sum( (true == 0) & (pred == 1))
    tn <- sum( (true == 1) & (pred == 0))
    p <- tp / (tp + fp); r <- tp / (tp + fn)
    f05 <- (1 + 0.25) * (p * r) / (0.25 * p + r)
  }
}

################################################################################################
# load data
{
  dev_train_basic <- fread("data/dev_train_basic.csv",stringsAsFactors =FALSE)
  dev_test_basic <- fread("data/dev_test_basic.csv",stringsAsFactors =FALSE)
  cookie_all_basic <- fread("data/cookie_all_basic.csv",stringsAsFactors =FALSE)
  
 }

################################################################################################
# data processing
{
  keycols <- c("country", "anonymous_c0", "anonymous_c1", "anonymous_c2")
  xsummary <- ddply(cookie_all_basic, .(country, anonymous_c0, anonymous_c1, anonymous_c2), .fun=myFun)
  xsummary <- xsummary[,c(keycols, "cookie_id")]
  xsummary <- unique(xsummary)
  xsummary <- data.table(xsummary)
  setkeyv(xsummary,keycols)
  
  
  setkeyv(dev_test_basic,keycols)
  setkeyv(dev_train_basic,keycols)
  
  xtrain <- merge(dev_train_basic,xsummary,all.x=TRUE)
  xtest <- merge(dev_test_basic,xsummary,all.x=TRUE)
  
  xsub <- data.frame(xtest)[,c("device_id","cookie_id")]
   write.csv(xsub,file="submissions/submission_country_c0_c1_c2_popular_cookie.csv",
            row.names=FALSE, quote = F)
  
  
  
}







xsub <- merge(dev_test_basic,country_anon5_summary,all.x=TRUE)
xsub <- data.frame(xtest)
xsub <- xsub[,c("device_id","freq")]
names(xsub) <- c("device_id","cookie_id")
write.csv(xsub,file="submissions/submission_country_c0_popular_cookie.csv",
          row.names=FALSE, quote = F)


## 
country_summary <- ddply(cookie_all_basic, .(country), .fun=myFun)
country_summary <- country_summary[,c(5,12)]


setkey(dev_test_basic,country)
country_summary <- unique(country_summary)
country_summary <- data.table(country_summary)
setkey(country_summary,country)


submission_country_popular_cookie <- merge(dev_test_basic,country_summary,all.x=TRUE)
submission_country_popular_cookie <- data.frame(submission_country_popular_cookie)
submission_country_popular_cookie <- submission_country_popular_cookie[,c("device_id","freq")]

names(submission_country_popular_cookie) <- c("device_id","freq")
write.csv(submission_country_popular_cookie,file="submissions/submission_country_popular_cookie.csv",
          row.names=FALSE, quote = F)
