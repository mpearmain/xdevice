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
    x$freq <- rep(names(tbl)[which.max(tbl)],nrow(x))
    x
  }
  
  # print a formatted message
  msg <- function(mmm,...)
  {
    cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
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
# load data: id_all_property
{
  id_all_ip <- readLines('./data/id_all_property.csv')
}

################################################################################################
# load data: id_all_property
{
#   dev_train_basic <- fread("data/dev_train_basic.csv",stringsAsFactors =FALSE)
#   dev_test_basic <- fread("data/dev_test_basic.csv",stringsAsFactors =FALSE)
#   cookie_all_basic <- fread("data/cookie_all_basic.csv",stringsAsFactors =FALSE)
  
  ## id_all_property
  # processed version obtained via ./command_line/basic.sh
  id_all_property <- fread('./data/id_all_property_no_count.csv', header = T, sep = ";")
  setnames(id_all_property, colnames(id_all_property), c(colnames(id_all_property)[1:2],"property_list"))
  
  # property list
  propList_full <- unlist(str_split(id_all_property$property_list, ","))
  propList_unique <- unique(propList_full); xf <- factor(propList_full)
  xtab <- table(xf)
  
  # grab the 20 most frequently occurring properties
  loc_list <- names(tail(sort(xtab), n = 20))
  id_combined <- array(0, c(nrow(id_all_property), length(loc_list)))
  for (ii in seq(loc_list))
  {
    idx <- grep(pattern = loc_list[ii], x = id_all_property$property_list)
    id_combined[idx,ii] <- 1
    msg(ii)
  }
  colnames(id_combined) <- loc_list
  
  # attach to the original
  id_combined <- data.frame(id_all_property, id_combined)
  id_combined$property_list <- NULL
  
  # store
  write.table(id_combined, file = "./data/id_property_top20_noCounts.csv",
              row.names = F, col.names = T, sep = ",", quote = F)
 }
