# Initial method of cutting the training set to have a validation set.
library(data.table)

full.train <- fread("./data/dev_train_basic.csv")
# Lets cut 10% randomly
set.seed(1401081)
full.train[, rnd := runif(dim(full.train)[1])]

train <- full.train[rnd > 0.1, ]
validation <- full.train[rnd <= 0.1, ]

train[, rnd := NULL]
validation[, rnd := NULL]

write.csv(train, './data/train_basic90.csv', row.names = F, quote = F)
write.csv(validation, './data/valid_basic10.csv', row.names = F, quote = F)  