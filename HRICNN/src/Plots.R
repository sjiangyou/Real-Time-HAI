library(here)
library(ggplot2)
library(grid)

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
# Files beginning with "rain_" represent rainfall-only models. Files beginning
# with "rain_ships_" represent rainfall-plus-SHIPS models. Each input mode is
# saved as one six-panel image containing both stages for all three models.
# This script writes only these three plot files:
#   - Plots/hyperparameter_selection_Rain.png
#   - Plots/hyperparameter_selection_RainShips.png
#   - Plots/hricnn_model_comparisons.png
#   - Plots/model_metric_comparisons.png

results_directory <- here::here("HRICNN", "Results")
plots_directory <- file.path(results_directory, "Plots")
dir.create(plots_directory, recursive = TRUE, showWarnings = FALSE)

model_names <- c("Model1", "Model2", "Model3")

input_modes <- list(
    Rain = list(prefix = "rain_", label = "RAIN"),
    RainShips = list(prefix = "rain_ships_", label = "RAIN+SHIPS")
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

# Arrange the three stage 1 plots above the three stage 2 plots without adding
# a plotting-package dependency such as patchwork or gridExtra. Returning a
# grob allows ggsave() to save the complete six-panel figure directly.
combine_hyperparameter_plots <- function(
    stage_one_plots,
    stage_two_plots
) {
    if (length(stage_one_plots) != length(stage_two_plots)) {
        stop("Stage 1 and stage 2 plot lists must have the same length.")
    }

    plot_grobs <- lapply(
        c(stage_one_plots, stage_two_plots),
        ggplotGrob
    )
    plot_viewports <- lapply(
        seq_along(plot_grobs),
        function(index) {
            grid::viewport(
                layout.pos.row = if (index <= length(stage_one_plots)) 1 else 2,
                layout.pos.col = ((index - 1) %% length(stage_one_plots)) + 1
            )
        }
    )

    grid::gTree(
        children = do.call(
            grid::gList,
            lapply(
                seq_along(plot_grobs),
                function(index) {
                    grid::editGrob(
                        plot_grobs[[index]],
                        vp = plot_viewports[[index]]
                    )
                }
            )
        ),
        vp = grid::viewport(layout = grid::grid.layout(2, 3))
    )
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

    stage_one_plots <- lapply(
        model_names,
        function(model_name) {
            plot_stage_one(stage_one_losses, model_name, mode$label)
        }
    )
    stage_two_plots <- lapply(
        model_names,
        function(model_name) {
            plot_stage_two(stage_two_losses, model_name, mode$label)
        }
    )

    ggsave(
        filename = file.path(
            plots_directory,
            paste0("hyperparameter_selection_", mode_name, ".png")
        ),
        plot = combine_hyperparameter_plots(stage_one_plots, stage_two_plots),
        width = 14,
        height = 10,
        dpi = 300
    )
}

# Compare all six trained HRICNN CNN variants. Each metric facet contains
# three model groups, with adjacent RAIN and RAIN+SHIPS bars.
plot_hricnn_model_comparisons <- function(
    metric_csv_path = file.path(results_directory, "model_metrics.csv"),
    output_path = file.path(plots_directory, "hricnn_model_comparisons.png")
) {
    if (!file.exists(metric_csv_path)) {
        stop("Missing model-metrics file: ", metric_csv_path)
    }

    metrics <- read.csv(
        metric_csv_path,
        stringsAsFactors = FALSE,
        check.names = FALSE,
        row.names = 1
    )
    metric_names <- c("POD", "FAR", "PSS", "BSS", "OS")
    missing_metrics <- setdiff(metric_names, names(metrics))
    if (length(missing_metrics) > 0) {
        stop(
            "The file ", metric_csv_path, " is missing metric column(s): ",
            paste(missing_metrics, collapse = ", ")
        )
    }

    model_ids <- c(
        paste0("RAIN_Model", 1:3),
        paste0("RAIN+SHIPS_Model", 1:3)
    )
    missing_model_ids <- setdiff(model_ids, rownames(metrics))
    if (length(missing_model_ids) > 0) {
        stop(
            "The model-metrics file is missing CNN row(s): ",
            paste(missing_model_ids, collapse = ", ")
        )
    }

    metric_values <- do.call(
        rbind,
        lapply(metric_names, function(metric_name) {
            values <- as.numeric(metrics[model_ids, metric_name])
            if (anyNA(values)) {
                stop("Metric column ", metric_name, " contains non-numeric values.")
            }

            data.frame(
                Model = rep(paste0("Model ", 1:3), times = 2),
                Mode = rep(c("RAIN", "RAIN+SHIPS"), each = 3),
                Metric = metric_name,
                Value = values,
                stringsAsFactors = FALSE
            )
        })
    )
    metric_values$Metric <- factor(metric_values$Metric, levels = metric_names)
    metric_values$Model <- factor(
        metric_values$Model,
        levels = paste0("Model ", 1:3)
    )
    metric_values$Mode <- factor(
        metric_values$Mode,
        levels = c("RAIN", "RAIN+SHIPS")
    )

    comparison_plot <- ggplot(
        metric_values,
        aes(x = Model, y = Value, fill = Mode)
    ) +
    geom_col(
        position = position_dodge(width = 0.8),
        width = 0.7
    ) +
    geom_hline(yintercept = 0, color = "grey50", linewidth = 0.3) +
    facet_wrap(~Metric, ncol = 2, nrow = 3) +
    scale_fill_manual(
        values = c(
            "RAIN" = "#d73027",
            "RAIN+SHIPS" = "#2166ac"
        )
    ) +
    labs(
        title = "HRICNN model test-metric comparison",
        x = "Model",
        y = "Metric value",
        fill = "Input mode"
    ) +
    theme_minimal(base_size = 12) +
    theme(
        legend.position = "bottom",
        strip.text = element_text(face = "bold")
    )

    ggsave(
        filename = output_path,
        plot = comparison_plot,
        width = 12,
        height = 10,
        dpi = 300
    )

    invisible(comparison_plot)
}

plot_hricnn_model_comparisons()

# Create one figure containing the five requested metrics for the selected
# HRICNN, SHIPS, and any additional comparison models in model_metrics.csv.
# The best RAIN+SHIPS CNN is selected by OS and displayed as HRICNN. The other
# internal RAIN/RAIN+SHIPS CNN rows are omitted from this final comparison.
plot_metric_comparisons <- function(
    metric_csv_path = file.path(results_directory, "model_metrics.csv"),
    output_path = file.path(plots_directory, "model_metric_comparisons.png")
) {
    if (!file.exists(metric_csv_path)) {
        stop("Missing model-metrics file: ", metric_csv_path)
    }

    metrics <- read.csv(
        metric_csv_path,
        stringsAsFactors = FALSE,
        check.names = FALSE,
        row.names = 1
    )
    metric_names <- c("POD", "FAR", "PSS", "BSS", "OS")
    missing_metrics <- setdiff(metric_names, names(metrics))

    if (length(missing_metrics) > 0) {
        stop(
            "The file ", metric_csv_path, " is missing metric column(s): ",
            paste(missing_metrics, collapse = ", ")
        )
    }

    if (nrow(metrics) == 0) {
        stop("The model-metrics file contains no model rows: ", metric_csv_path)
    }

    model_ids <- rownames(metrics)
    if (any(is.na(model_ids) | model_ids == "")) {
        stop("Every model row in the model-metrics file must have a name.")
    }

    rain_ships_model_ids <- grep(
        "^RAIN\\+SHIPS_Model[0-9]+$",
        model_ids,
        value = TRUE
    )
    if (length(rain_ships_model_ids) == 0) {
        stop(
            "The model-metrics file must contain at least one RAIN+SHIPS CNN row."
        )
    }

    rain_ships_os <- as.numeric(metrics[rain_ships_model_ids, "OS"])
    if (anyNA(rain_ships_os)) {
        stop("RAIN+SHIPS CNN OS values must be numeric.")
    }
    best_rain_ships_id <- rain_ships_model_ids[which.max(rain_ships_os)]

    if (!("SHIPS" %in% model_ids)) {
        stop("The model-metrics file must contain a SHIPS row.")
    }

    internal_cnn_ids <- grep(
        "^(RAIN|RAIN\\+SHIPS)_Model[0-9]+$",
        model_ids,
        value = TRUE
    )
    additional_model_ids <- setdiff(model_ids, c(internal_cnn_ids, "SHIPS"))
    comparison_ids <- c(
        best_rain_ships_id,
        "SHIPS",
        additional_model_ids
    )
    comparison_names <- comparison_ids
    comparison_names[comparison_names == best_rain_ships_id] <- "HRICNN"
    comparison_names[comparison_names == "SHIPS"] <- "SHIPS-RII"

    model_colors <- c(
        HRICNN = "#2166ac",
        "SHIPS-RII" = "#3b3b75"
    )
    if (length(additional_model_ids) > 0) {
        additional_colors <- grDevices::hcl.colors(
            length(additional_model_ids),
            palette = "Dark 3"
        )
        names(additional_colors) <- additional_model_ids
        model_colors <- c(model_colors, additional_colors)
    }

    metric_values <- do.call(
        rbind,
        lapply(metric_names, function(metric_name) {
            values <- as.numeric(metrics[comparison_ids, metric_name])
            if (anyNA(values)) {
                stop("Metric column ", metric_name, " contains non-numeric values.")
            }

            data.frame(
                Model = comparison_names,
                Metric = metric_name,
                Value = values,
                stringsAsFactors = FALSE
            )
        })
    )
    metric_values$Metric <- factor(metric_values$Metric, levels = metric_names)
    metric_values$Model <- factor(
        metric_values$Model,
        levels = comparison_names
    )

    comparison_plot <- ggplot(
        metric_values,
        aes(x = Model, y = Value, fill = Model)
    ) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0, color = "grey50", linewidth = 0.3) +
    facet_wrap(~Metric, ncol = 5) +
    scale_fill_manual(values = model_colors) +
    labs(
        title = "HRICNN test-metric comparison",
        x = "Model",
        y = "Metric value",
        fill = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
        legend.position = "bottom",
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1)
    )

    ggsave(
        filename = output_path,
        plot = comparison_plot,
        width = 18,
        height = 5.5,
        dpi = 300
    )

    invisible(comparison_plot)
}

plot_metric_comparisons()

message("Saved HRICNN plots to: ", plots_directory)
