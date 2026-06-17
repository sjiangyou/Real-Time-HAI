here::i_am("HIECNN/CODE/Model_Scatterplots.R")
library(here)
setwd(here("HIECNN"))
options(scipen = 999)

analyze <- read.csv("OUTPUT_VMAX_IMERG/")
analyze$category <- ifelse(analyze$VMAX >= 96, "Maj",
                        ifelse(analyze$VMAX >= 64, "Min",
                            ifelse(analyze$VMAX >= 34, "TS", "TD")))
analyze$color <- ifelse(analyze$category == "TD", "black",
                    ifelse(analyze$category == "TS", "green",
                        ifelse(analyze$category == "Min", "red", "blue")))

analyzeTD <- analyze[analyze$category == "TD", ]
analyzeTS <- analyze[analyze$category == "TS", ]
analyzeMin <- analyze[analyze$category == "Min", ]
analyzeMaj <- analyze[analyze$category == "Maj", ]

analyzeTD <- na.omit(analyzeTD)
analyzeTS <- na.omit(analyzeTS)
analyzeMin <- na.omit(analyzeMin)
analyzeMaj <- na.omit(analyzeMaj)

MAE_All <- mean(abs(analyze$VMAX-analyze$preds))
MAE_TD <- mean(abs(analyzeTD$VMAX-analyzeTD$preds))
MAE_TS <- mean(abs(analyzeTS$VMAX-analyzeTS$preds))
MAE_Min <- mean(abs(analyzeMin$VMAX-analyzeMin$preds))
MAE_Maj <- mean(abs(analyzeMaj$VMAX-analyzeMaj$preds))

RMSE_All <- sqrt(mean((analyze$VMAX-analyze$preds)^2))
RMSE_TD <- sqrt(mean((analyzeTD$VMAX-analyzeTD$preds)^2))
RMSE_TS <- sqrt(mean((analyzeTS$VMAX-analyzeTS$preds)^2))
RMSE_Min <- sqrt(mean((analyzeMin$VMAX-analyzeMin$preds)^2))
RMSE_Maj <- sqrt(mean((analyzeMaj$VMAX-analyzeMaj$preds)^2))

All <- nrow(analyze)
TD <- nrow(analyzeTD)
TS <- nrow(analyzeTS)
Min <- nrow(analyzeMin)
Maj <- nrow(analyzeMaj)

dir.create("FIGURES", recursive = TRUE, showWarnings = FALSE)
png("FIGURES/Linear_Scatterplot.png", width = 600, height = 600)

rsq <- round(cor(analyze$VMAX, analyze$preds)^2, 2)
plot(analyze$preds, analyze$VMAX,
    xlim = c(15, 160), ylim = c(15, 160), col = analyze$color, pch = 18,
    xlab = "Model Estimated Vmax (kt)", ylab = "Actual Vmax (kt)",
    main = "Vmax Only",
    cex.main = 1.6, cex.lab = 1.6, cex.axis = 1.6, cex.sub = 1.6)
abline(a = 0, b = 1)
legend("bottomright", legend = c(paste0("TD", " (", TD, ")"), paste0("TS", " (", TS, ")"), paste0("CAT12", " (", Min, ")"), paste0("CAT35", " (", Maj, ")")), 
       fill = c("black", "green", "red", "blue"), ncol = 2, cex = 1.4)
text(x = 40, y = 150, paste0("MAE: ", round(MAE_All, 2), " kt"), cex = 1.4)
text(x = 40, y = 140, paste0("RMSE: ", round(RMSE_All, 2), " kt"), cex = 1.4)
text(x = 40, y = 130, bquote(R^2 == .(rsq)), cex = 1.4)
dev.off()
