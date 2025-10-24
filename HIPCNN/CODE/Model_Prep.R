# The best use of this script is after running the Add_Future_Land_Ocean.ipynb script.

setwd("~/Projects/Real-Time-HAI/HIPCNN/IMERG/")
options(scipen = 999)

data <- read.csv("Land_Ocean_Futures.csv", header = TRUE)
data <- data[, -1]
data <- data[data$BASIN == 'ATL',]

columns <- c('GIS_ID', 'DATE', 'STORM_NAME', 'STORM_ID', 'LAND_OCEAN','SHIPS_PER', 
             'SHIPS_POT', 'VMAX')

data_structure <- data.frame(time=c("P06","P12","P18","P24"), 
                             value=c("_2","_3","_4","_5"))

doresample <- function(d, c, n){
  t <- d[d$Category == c, ]
  if(nrow(t) < n){
    t2 <- t[sample(1:nrow(t), n-nrow(t), replace = T),]
    t <- rbind(t,t2)
  }
  return(t)
}

# Predictions

for (i in 1:nrow(data_structure)) {
  x <- data[,c(columns, paste0("VMAX_", data_structure$time[i]), grep(paste0("SHDC", data_structure$value[i]), names(data), value = T), paste0("LO", data_structure$time[i]))]
  
  x[x == -999] <- NA
  x <- x[complete.cases(x), ]
  names(x) <- gsub(data_structure$value[i], "", names(x))
  names(x) <- gsub(data_structure$time[i], "FT", names(x))
  x <- x[x$LOFT == 'Ocean', ]
  x <- x[x$LAND_OCEAN == "Ocean", ]
  x$IC <- x$VMAX_FT - x$VMAX
  x$Category <- ifelse(x$VMAX <= 33, "TD", 
                       ifelse(x$VMAX <= 63, "TS", 
                              ifelse(x$VMAX <= 95, "Min", "Maj")))

  assign(paste0(data_structure$time[i], "_train"), x)
  
  x_train_resample <- do.call(rbind, lapply(unique(x$Category), function(y) doresample(x,y, max(table(x$Category)))))
  assign(paste0(data_structure$time[i], "_train_resample"), x_train_resample)
  
  write.csv(x_train_resample, file = paste0("IC_Data/ATL/", data_structure$time[i], '_train_resample.csv'))
}