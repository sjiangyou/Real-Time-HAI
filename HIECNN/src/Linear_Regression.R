here::i_am("HIECNN/CODE/Linear_Regression.R")
library(here)
setwd(here())
options(scipen = 999)

train <- read.csv("HIECNN/IMERG/DEV/ALL_TRAIN_DATA.csv")
model <- lm(VMAX ~ VMAX_N12 + VMAX_N06, data = train)
print(summary(model))

test <- read.csv("HIECNN/IMERG/DEV/ALL_TEST_DATA.csv")
test$preds <- predict(model, newdata = test)
dir.create("HIECNN/OUTPUT_VMAX", recursive = TRUE, showWarnings = FALSE)
write.csv(test, "HIECNN/OUTPUT_VMAX/MULTILINEAR_RESULTS.csv", row.names = FALSE)
