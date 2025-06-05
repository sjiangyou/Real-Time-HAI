here::i_am("HIECNN/CODE/Model_Prep.R")
library(here)
setwd(here())
options(scipen = 999)

main <- function() {
    train_end <- 2018
    train_end <- as.numeric(paste0(train_end, "000000"))

    all_data <- process_data()
    train <- all_data[all_data$DATE < train_end, ]
    test <- all_data[all_data$DATE >= train_end, ]

    train$CAT <- ifelse(train$VMAX <= 33, "TD",
        ifelse(train$VMAX <= 63, "TS",
            ifelse(train$VMAX <= 95, "Min", "Maj")))

    test$CAT <- ifelse(test$VMAX <= 33, "TD",
        ifelse(test$VMAX <= 63, "TS",
            ifelse(test$VMAX <= 95, "Min", "Maj")))

    train_resample <- do.call(rbind,
        lapply(unique(train$CAT),
            function(y) doresample(train, y, max(table(train$CAT)))))

    print(paste0("Training data size: ", nrow(train)))
    print(paste0("Testing data size: ", nrow(test)))

    dir.create("HIECNN/IMERG/DEV", recursive = TRUE, showWarnings = FALSE)

    write.csv(train, "HIECNN/IMERG/DEV/ALL_TRAIN_DATA.csv", row.names = FALSE)
    write.csv(test, "HIECNN/IMERG/DEV/ALL_TEST_DATA.csv", row.names = FALSE)
    write.csv(train_resample, "HIECNN/IMERG/DEV/ALL_TRAIN_DATA_RESAMPLE.csv", row.names = FALSE)
}

process_data <- function() {
    all_data <- read.csv("BRTK_SHIPS_2000to2019_IMERG_OK_Request_2023_FINAL.csv")
    all_data <- all_data[, 1:14]
    all_data[all_data == -999] <- NA
    all_data <- all_data[complete.cases(all_data), ]
    return(all_data)
}

doresample <- function(data, category, number) {
    t <- data[data$CAT == category, ]
    if (nrow(t) < number) {
        t2 <- t[sample(seq_len(nrow(t)), number - nrow(t), replace = TRUE), ]
        t <- rbind(t, t2)
    }
    return(t)
}

main()
