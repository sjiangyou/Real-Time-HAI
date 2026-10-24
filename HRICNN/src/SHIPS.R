library(here)
library(MASS)

setwd(here::here("HRICNN"))
train <- read.csv("IMERG/ATL_train.csv")
train_resample <- read.csv("IMERG/ATL_train_resample.csv")
test <- read.csv("IMERG/ATL_test.csv")

ships_formula <- RI ~ VMAX + SHIPS_PER + SHIPS_POT_Avg24h +
    SHIPS_NOHC_Avg24h + SHIPS_SHDC_Avg24h + SHIPS_CFLX_Avg24h +
    SHIPS_D200_Avg24h + SHIPS_MTPW_108h + SHIPS_PC2 + SHIPS_IR00_12h

write_ships_predictions <- function(training_data, test_data, output_path) {
    model <- lda(ships_formula, data = training_data)
    predictions <- predict(model, newdata = test_data)$posterior[, 2]
    output <- test_data
    output$Prediction <- predictions
    write.csv(output, file = output_path, row.names = FALSE)
}

# Both baselines are evaluated on the same untouched test set. Resampling only
# changes the training data used to fit the LDA model.
write_ships_predictions(
    train,
    test,
    "Results/SHIPS_predictions.csv"
)
write_ships_predictions(
    train_resample,
    test,
    "Results/SHIPS_predictions_resample.csv"
)
