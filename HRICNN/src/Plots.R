library(here)
library(ggplot2)

# Plot the staged HRICNN hyperparameter-selection results.
#
# Stage 1 evaluates every batch size through epoch 10, so its plots show the
# maximum OS reached by each batch size. Stage 2 evaluates the selected batch
# size for each architecture over all candidate epochs, so its plots show
# every epoch and highlight the single maximum-OS epoch in red.
#
# Run from the repository root with:
#   Rscript HRICNN/src/Plots.R
#
# Standard files represent models trained with SHIPS data (Rain). Files whose
# names begin with "rain_" represent image-only models without SHIPS data.

results_directory <- here::here("HRICNN", "Results")
plots_directory <- file.path(results_directory, "Plots")
dir.create(plots_directory, recursive = TRUE, showWarnings = FALSE)

model_names <- c("Model1", "Model2", "Model3")

input_modes <- list(
    Rain = list(prefix = "", label = "Rain + SHIPS"),
    NoRain = list(prefix = "rain_", label = "Rain only")
)

read_loss_data <- function(path) {
    if (!file.exists(path)) {
        stop("Missing hyperparameter-loss file: ", path)
    }

    losses <- read.csv(path, stringsAsFactors = FALSE)
    required_columns <- c("Model", "Epoch", "Batch", "Loss", "OS")
    missing_columns <- setdiff(required_columns, names(losses))
    if (length(missing_columns) > 0) {
        stop(
            "The file ", path, " is missing columns: ",
            paste(missing_columns, collapse = ", ")
        )
    }

    losses$Model <- as.character(losses$Model)
    losses$Epoch <- as.integer(losses$Epoch)
    losses$Batch <- as.integer(losses$Batch)
    losses$Loss <- as.numeric(losses$Loss)
    losses$OS <- as.numeric(losses$OS)
    losses
}

maximum_by_batch <- function(losses, model_name) {
    model_losses <- losses[losses$Model == model_name, , drop = FALSE]
    if (nrow(model_losses) == 0) {
        stop("No stage 1 results found for ", model_name)
    }

    batches <- sort(unique(model_losses$Batch))
    do.call(
        rbind,
        lapply(batches, function(batch_size) {
            batch_losses <- model_losses[model_losses$Batch == batch_size, , drop = FALSE]
            maximum_row <- batch_losses[which.max(batch_losses$OS), , drop = FALSE]
            data.frame(
            Batch = batch_size,
            MaximumOS = maximum_row$OS,
            Epoch = maximum_row$Epoch
            )
        })
    )
}

plot_stage_one <- function(losses, model_name, mode_label) {
    summary <- maximum_by_batch(losses, model_name)
    summary$BatchLabel <- factor(summary$Batch, levels = summary$Batch)
    display_model_name <- sub("^Model", "Model ", model_name)
    summary$Bar <- "Other batch sizes"
    summary$Bar[which.max(summary$MaximumOS)] <- "Maximum-OS batch size"
    summary$Bar <- factor(
        summary$Bar,
        levels = c("Other batch sizes", "Maximum-OS batch size")
    )

    ggplot(summary, aes(x = BatchLabel, y = MaximumOS, fill = Bar)) +
    geom_col() +
    scale_fill_manual(
        values = c(
        "Other batch sizes" = "#2166ac",
        "Maximum-OS batch size" = "#d73027"
        )
    ) +
    labs(
        title = paste("Stage 1:", display_model_name, "batch-size selection"),
        subtitle = mode_label,
        x = "Batch size",
        y = "Maximum validation OS",
        fill = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
}

plot_stage_two <- function(losses, model_name, mode_label) {
    model_losses <- losses[losses$Model == model_name, , drop = FALSE]
    if (nrow(model_losses) == 0) {
        stop("No stage 2 results found for ", model_name)
    }
    display_model_name <- sub("^Model", "Model ", model_name)

# Use one red point when tied maxima occur, making the selected epoch
# unambiguous in the figure.
    maximum_index <- which.max(model_losses$OS)
    model_losses$Point <- "Other epochs"
    model_losses$Point[maximum_index] <- "Maximum-OS epoch"
    model_losses$Point <- factor(
        model_losses$Point,
        levels = c("Other epochs", "Maximum-OS epoch")
    )

    ggplot(model_losses, aes(x = Epoch, y = OS, color = Point)) +
    geom_point(size = 2.5) +
    scale_color_manual(
    values = c(
        "Other epochs" = "#2166ac",
        "Maximum-OS epoch" = "#d73027"
        )
    ) +
    labs(
        title = paste("Stage 2:", display_model_name, "epoch selection"),
        subtitle = mode_label,
        x = "Epoch",
        y = "Validation OS",
        color = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
}

for (mode_name in names(input_modes)) {
    mode <- input_modes[[mode_name]]
    stage_one_path <- file.path(
        results_directory,
        paste0(mode$prefix, "hyperparameter_validation_losses_stage1.csv")
    )
    stage_two_path <- file.path(
        results_directory,
        paste0(mode$prefix, "hyperparameter_validation_losses_stage2.csv")
    )

    stage_one_losses <- read_loss_data(stage_one_path)
    stage_two_losses <- read_loss_data(stage_two_path)

    for (model_name in model_names) {
        ggsave(
            filename = file.path(
                plots_directory,
                paste0("stage1_", mode_name, "_", model_name, ".png")
            ),
            plot = plot_stage_one(stage_one_losses, model_name, mode$label),
            width = 7,
            height = 5,
            dpi = 300
        )

        ggsave(
            filename = file.path(
                plots_directory,
                paste0("stage2_", mode_name, "_", model_name, ".png")
            ),
            plot = plot_stage_two(stage_two_losses, model_name, mode$label),
            width = 7,
            height = 5,
            dpi = 300
        )
    }
}

message("Saved HRICNN hyperparameter plots to: ", plots_directory)
