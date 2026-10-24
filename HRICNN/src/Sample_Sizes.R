library(here)

here::i_am("HRICNN/src/Sample_Sizes.R")
setwd(here::here("HRICNN"))

train <- read.csv("IMERG/ATL_train.csv")
test <- read.csv("IMERG/ATL_test.csv")

train_RI <- train[train$RI == 1, ]
test_RI <- test[test$RI == 1, ]

print(paste("Training set size:", nrow(train)))
print(paste("Training set RI size:", nrow(train_RI)))
print(paste("Training set RI storms", length(unique(train_RI$STORM_ID))))
print(paste("Training set non-RI size:", nrow(train) - nrow(train_RI)))
print(paste("Training set non-RI storms", length(unique(train$STORM_ID)) - length(unique(train_RI$STORM_ID))))
print(paste("Testing set size:", nrow(test)))
print(paste("Testing set RI size:", nrow(test_RI)))
print(paste("Testing set RI storms", length(unique(test_RI$STORM_ID))))
print(paste("Testing set non-RI size:", nrow(test) - nrow(test_RI)))
print(paste("Testing set non-RI storms", length(unique(test$STORM_ID)) - length(unique(test_RI$STORM_ID))))
