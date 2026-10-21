library(here)
library(MASS)

setwd(here::here("HRICNN"))
train <- read.csv("IMERG/ATL_train.csv")
test <- read.csv("IMERG/ATL_test.csv")

ships <- lda(RI ~ VMAX + SHIPS_PER + SHIPS_POT_Avg24h + SHIPS_NOHC_Avg24h + SHIPS_SHDC_Avg24h + SHIPS_CFLX_Avg24h + SHIPS_D200_Avg24h + SHIPS_MTPW_108h + SHIPS_PC2 + SHIPS_IR00_12h, data = train)

predictions <- predict(ships, newdata = test)
predictions <- predictions$posterior[, 2]
test$Prediction<- predictions
write.csv(test, file = "Results/SHIPS_predictions.csv")
