# Reading Bad CSV Files
# Unfortunately, three of the data files are improperly formatted CSV files.
# They have a variable-length field in them separated by the { and } braces that contain an array of tuples.
library(readr)

# function read_bad_csv reads these CSV files into a proper relational table
# (which dramatically increases the number of rows).

read_bad_csv <- function(file_name, bad_col=3, n_max=-1) {
  f_in <- file(file_name)
  lines <- readLines(f_in, n=n_max)
  close(f_in)
  temp_csv_1 <- tempfile()
  f_out_1 <- file(temp_csv_1, "w")
  writeLines(gsub("\\{|\\}", '"', lines), f_out_1)
  close(f_out_1)
  data <- read_csv(temp_csv_1, col_names=FALSE)
  temp_csv_2 <- tempfile()
  f_out_2 <- file(temp_csv_2, "w")
  for (i in 1:nrow(data)) {
    bad_lines <- strsplit(substr(data[i,bad_col], 2, nchar(data[i,bad_col])-1), "\\),\\(")[[1]]
    if (bad_col==1) {
      lines <- paste(bad_lines,
                     paste0(as.character(data[i,2:ncol(data)]), collapse=","),
                     sep=",")
    } else if (bad_col<ncol(data)) {
      lines <- paste(paste0(as.character(data[i,1:bad_col-1]), collapse=","),
                     bad_lines,
                     paste0(as.character(data[i,bad_col+1:ncol(data)]), collapse=","),
                     sep=",")
    } else {
      lines <- paste(paste0(as.character(data[i,1:ncol(data)-1]), collapse=","),
                     bad_lines,
                     sep=",")
    }
    writeLines(lines, f_out_2)
  }
  close(f_out_2)
  return(read_csv(temp_csv_2))
}

# Here, we'll read it in properly (set n_max to -1 to process all the lines).
# property_category.csv.
property_category <- read_bad_csv("./data/property_category.csv", bad_col=2, n_max=-1)
write.csv(property_category, './data/property_category_DF.csv', row.names = F)

ip <- read_bad_csv("./data/id_all_ip.csv", bad_col=3, n_max=-1)
write.csv(ip, './data/id_all_ip_DF.csv', row.names = F)

#The next improperly formatted CSV file is id_all_property.csv.
property <- read_bad_csv("./data/id_all_property.csv", bad_col=3, n_max=-1)
write.csv(property, './data/id_all_property_DF.csv', row.names = F)
