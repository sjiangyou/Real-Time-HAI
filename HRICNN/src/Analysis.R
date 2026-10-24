# Calculate probability-based metrics for HRICNN test predictions.
#
# Run from the repository root with:
#   Rscript HRICNN/src/Analysis.R
#
# The script intentionally keeps each metric in its own function. To add a
# metric, define a function that accepts observed labels and predicted
# probabilities, then add its result to calculate_model_metrics().

library(here)

# Classification boundary for future threshold-based metrics. This does not
# affect the probability-based Brier Score or Brier Skill Score.
classification_boundary <- 0.21


classify_predictions <- function(predicted, boundary = classification_boundary) {
    if (length(boundary) != 1 || is.na(boundary) ||
        boundary < 0 || boundary > 1) {
        stop("classification boundary must be a single value between 0 and 1.")
    }

    if (anyNA(predicted) || any(predicted < 0 | predicted > 1)) {
        stop("predicted probabilities must be between 0 and 1.")
    }

    ifelse(predicted >= boundary, 1, 0)
}

calculate_POD <- function(observed, predicted) {
    true_positives <- sum(observed == 1 & predicted == 1)
    false_negatives <- sum(observed == 1 & predicted == 0)

    if (true_positives + false_negatives == 0) {
        return(NA)
    }

    true_positives / (true_positives + false_negatives)
}

calculate_FAR <- function(observed, predicted) {
    false_positives <- sum(observed == 0 & predicted == 1)
    true_positives <- sum(observed == 1 & predicted == 1)

    if (false_positives + true_positives == 0) {
        return(NA)
    }

    false_positives / (false_positives + true_positives)
}

calculate_PSS <- function(observed, predicted) {
    true_positives <- sum(observed == 1 & predicted == 1)
    false_positives <- sum(observed == 0 & predicted == 1)
    false_negatives <- sum(observed == 1 & predicted == 0)
    true_negatives <- sum(observed == 0 & predicted == 0)

    numerator <- (true_positives * true_negatives) - (false_positives * false_negatives)
    denominator <- (true_positives + false_negatives) * (false_positives + true_negatives)

    if (denominator == 0) {
        return(NA)
    }

    numerator / denominator
}

calculate_brier_score <- function(observed, predicted) {
    mean((observed - predicted)^2)
}


calculate_climatological_brier_score <- function(
    observed,
    climatological_probability = mean(observed)
) {
    if (length(climatological_probability) != 1 ||
        is.na(climatological_probability) ||
        climatological_probability < 0 || climatological_probability > 1) {
        stop("climatological probability must be a single value between 0 and 1.")
    }

    mean((observed - climatological_probability)^2)
}


calculate_brier_skill_score <- function(
    observed,
    predicted,
    climatological_probability = mean(observed)
) {
    brier_score <- calculate_brier_score(observed, predicted)
    climatological_brier_score <- calculate_climatological_brier_score(
        observed,
        climatological_probability
    )

    if (climatological_brier_score == 0) {
        stop(
            "The climatological Brier Score is zero; ",
            "the Brier Skill Score is undefined."
        )
    }

    1 - (brier_score / climatological_brier_score)
}


calculate_model_metrics <- function(data, climatological_probability) {
    predicted_classes <- classify_predictions(data$Prediction)

    metrics <- list(
        brier_score = calculate_brier_score(data$RI, data$Prediction),
        brier_skill_score = calculate_brier_skill_score(
            data$RI,
            data$Prediction,
            climatological_probability
        ),
        pod = calculate_POD(data$RI, predicted_classes),
        far = calculate_FAR(data$RI, predicted_classes),
        pss = calculate_PSS(data$RI, predicted_classes)
    )

    metrics$overall_score <- metrics$brier_skill_score +
        metrics$pod - metrics$far + metrics$pss
    metrics
}


validate_prediction_data <- function(data, file_name) {
    required_columns <- c("GIS_ID", "RI", "Prediction")
    missing_columns <- setdiff(required_columns, names(data))

    if (length(missing_columns) > 0) {
        stop(
            file_name,
            " is missing required column(s): ",
            paste(missing_columns, collapse = ", ")
        )
    }

    if (anyNA(data[, required_columns])) {
        stop(file_name, " contains missing values in a required column.")
    }

    if (!all(data$RI %in% c(0, 1))) {
        stop(file_name, " contains RI values other than 0 and 1.")
    }

    if (!is.numeric(data$Prediction) ||
        any(data$Prediction < 0 | data$Prediction > 1)) {
        stop(
            file_name,
            " contains predictions outside the probability range [0, 1]."
        )
    }
}


validate_matching_observations <- function(first_data, second_data) {
    if (nrow(first_data) != nrow(second_data)) {
        stop("The model output files contain different numbers of observations.")
    }

    if (!identical(first_data$GIS_ID, second_data$GIS_ID) ||
        !identical(first_data$RI, second_data$RI)) {
        stop("The model output files do not contain matching observations.")
    }
}


print_model_metrics <- function(model_name, metrics) {
    cat(model_name, "\n", sep = "")
    cat(sprintf("  Brier Score:       %.6f\n", metrics$brier_score))
    cat(sprintf("  Brier Skill Score: %.6f\n", metrics$brier_skill_score))
    cat(sprintf("  POD:               %.6f\n", metrics$pod))
    cat(sprintf("  FAR:               %.6f\n", metrics$far))
    cat(sprintf("  PSS:               %.6f\n", metrics$pss))
    cat(sprintf("  OS:                %.6f\n", metrics$overall_score))
    metrics
}

save_all_model_metrics <- function(metrics_list, output_path) {
    metrics_df <- do.call(rbind, lapply(metrics_list, as.data.frame))
    rownames(metrics_df) <- names(metrics_list)
    colnames(metrics_df) <- c("BS", "BSS", "POD", "FAR", "PSS", "OS")
    write.csv(metrics_df, file = output_path, row.names = TRUE)
}


results_directory <- here::here("HRICNN", "Results")
ships_output_path <- file.path(results_directory, "SHIPS_predictions.csv")
ships_resample_output_path <- file.path(
    results_directory,
    "SHIPS_predictions_resample.csv"
)

cnn_output_paths <- c(
    "RAIN+SHIPS_Model1" = file.path(
        results_directory, "rain_ships_model_output_Model1.csv"
    ),
    "RAIN+SHIPS_Model2" = file.path(
        results_directory, "rain_ships_model_output_Model2.csv"
    ),
    "RAIN+SHIPS_Model3" = file.path(
        results_directory, "rain_ships_model_output_Model3.csv"
    ),
    "RAIN_Model1" = file.path(
        results_directory, "rain_model_output_Model1.csv"
    ),
    "RAIN_Model2" = file.path(
        results_directory, "rain_model_output_Model2.csv"
    ),
    "RAIN_Model3" = file.path(
        results_directory, "rain_model_output_Model3.csv"
    )
)

cnn_data <- lapply(cnn_output_paths, function(path) {
    read.csv(path, stringsAsFactors = FALSE)
})
names(cnn_data) <- names(cnn_output_paths)
ships_data <- read.csv(ships_output_path, stringsAsFactors = FALSE)
ships_resample_data <- read.csv(
    ships_resample_output_path,
    stringsAsFactors = FALSE
)

for (model_name in names(cnn_data)) {
    validate_prediction_data(cnn_data[[model_name]], model_name)
}
validate_prediction_data(ships_data, "SHIPS_predictions.csv")
validate_prediction_data(
    ships_resample_data,
    "SHIPS_predictions_resample.csv"
)
reference_data <- cnn_data[[1]]
for (model_name in names(cnn_data)[-1]) {
    validate_matching_observations(reference_data, cnn_data[[model_name]])
}
validate_matching_observations(reference_data, ships_data)
validate_matching_observations(reference_data, ships_resample_data)

climatological_probability <- 0.5
climatological_brier_score <- calculate_climatological_brier_score(
    reference_data$RI,
    climatological_probability
)

metrics_lines <- capture.output({
    cat("HRICNN prediction metrics\n")
    cat("=========================\n")
    cat(sprintf("Observations: %d\n", nrow(reference_data)))
    cat(sprintf("Reference probability: %.6f\n", climatological_probability))
    cat(sprintf("Test RI prevalence: %.6f\n", mean(reference_data$RI)))
    cat(sprintf(
        "Climatological Brier Score: %.6f\n\n",
        climatological_brier_score
    ))

    ships_metrics <- print_model_metrics(
        "SHIPS non-resampled",
        calculate_model_metrics(ships_data, climatological_probability)
    )
    ships_resample_metrics <- print_model_metrics(
        "SHIPS resampled",
        calculate_model_metrics(
            ships_resample_data,
            climatological_probability
        )
    )

    metrics_list <- list(
        "SHIPS_non_resampled" = ships_metrics,
        "SHIPS_resampled" = ships_resample_metrics
    )
    for (model_name in names(cnn_data)) {
        cat("\n")
        metrics_list[[model_name]] <- print_model_metrics(
            model_name,
            calculate_model_metrics(cnn_data[[model_name]], climatological_probability)
        )
    }
})
cat(paste(metrics_lines, collapse = "\n"), "\n", sep = "")
writeLines(metrics_lines, file.path(results_directory, "METRICS.txt"))

save_all_model_metrics(
    metrics_list,
    file.path(results_directory, "model_metrics.csv")
)
