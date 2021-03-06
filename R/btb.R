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
  
  # pick top 2 most popular cookies
  myFun2 <- function(x)
  {
    xtab <- data.frame(x)[,c("cookie_id", "nof_ips")]
    xtab <- xtab[ order(xtab$nof_ips, decreasing = T), ]
    top_cookies <- paste(xtab$cookie_id[1],  xtab$cookie_id[2]) 
    x$cookie_id <- rep(top_cookies,nrow(x))
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
  id_ip <- fread('./data/id_ip_counts.csv', stringsAsFactors = F)
  
}

################################################################################################
# data processing
{
  device_ip <- id_ip[device_or_cookie_indicator == 0,]
  device_ip$device_or_cookie_indicator <- NULL
  colnames(device_ip)[1] <- "device_id"
  cookie_ip <- id_ip[device_or_cookie_indicator == 1,]
  cookie_ip$device_or_cookie_indicator <- NULL
  colnames(cookie_ip)[1] <- "cookie_id"
  
  # attach cookie property info to cookie_all_basic
  cookie_all_basic <- merge(cookie_all_basic, cookie_ip, by = "cookie_id", all.x = T)
  cookie_all_basic[is.na(cookie_all_basic)] <- -1
  
  # attach device property to dev_train_basic
  dev_train_basic <- merge(dev_train_basic, device_ip, by = "device_id", all.x = T)
  dev_train_basic[is.na(dev_train_basic)] <- -1
  
  # attach device property to dev_test_basic
  dev_test_basic <- merge(dev_test_basic, device_ip, by = "device_id", all.x = T)
  dev_test_basic[is.na(dev_test_basic)] <- -1
  
  
}

################################################################################################
# build submission
{
  keycols <- c("country", "anonymous_c0", "anonymous_c1", "anonymous_c2")
  xsummary <- ddply(cookie_all_basic, 
                    .(country, anonymous_c0, anonymous_c1, anonymous_c2), 
                    .fun=myFun2)

  xsummary <- xsummary[,c(keycols, "cookie_id")]
  xsummary <- unique(xsummary)
  xsummary <- data.table(xsummary)
  setkeyv(xsummary,keycols)
  
  
  setkeyv(dev_test_basic,keycols)
  setkeyv(dev_train_basic,keycols)
  
  xtrain <- merge(dev_train_basic,xsummary,all.x=TRUE)
  xtest <- merge(dev_test_basic,xsummary,all.x=TRUE)
  
  xsub <- data.frame(xtest)[,c("device_id","cookie_id")]
   write.csv(xsub,file="submissions/sub_country_c0_c1_c2_2mostFreq.csv",
            row.names=FALSE, quote = F)

  
}
