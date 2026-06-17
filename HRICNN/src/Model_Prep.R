setwd("~/Documents/Real_Time_HAI/HRICNN")
data <- read.csv("BRTK_2000to2019_IMERG_SHIPS-RII.csv")
basin <-  ""

data <- data[, -1]

data_24_R <- data[, grepl(pattern = "Avg24", colnames(data))]
data_24_L <- data[, c(1,2,3,8,9,10,11,17,12,24,75,79,80)]
data_24 <- cbind(data_24_L, data_24_R)

for(i in 1:length(data_24$VMAX)){
  if(data_24$VMAX_P24[i] == -999 || data_24$VMAX[i] == -999){
    data_24$DVMAX_24[i] <- -999
  }
  else{
    data_24$DVMAX_24[i] <- data_24$VMAX_P24[i] - data_24$VMAX[i]
  }
}

data_24$DVMAX_24[data_24$DVMAX_24 == -999] <- NA
# dvmax <- data_24[, c(2,36)]
# dvmax <- dvmax[complete.cases(dvmax),]
data_24 <- data_24[complete.cases(data_24),]

data_24[data_24 == -999] <- NA
data_24 <- data_24[complete.cases(data_24),]

doresample <- function(d, c, n){
  t <- d[d$RI == c, ]
  if(nrow(t) < n){
    t2 <- t[sample(1:nrow(t), n-nrow(t), replace = T),]
    t <- rbind(t,t2)
  }
  return(t)
}

data_24$Category <- ifelse(data_24$VMAX <= 33, "TD", 
                             ifelse(data_24$VMAX <= 63, "TS", 
                                    ifelse(data_24$VMAX <= 95, "Min", "Maj")))
data_24$RI <- ifelse(data_24$DVMAX_24 >= 30, 1, 0)

for(b in unique(data_24$BASIN)){
  basindata <- data_24[data_24$BASIN == b,]
  train_resample <- do.call(rbind, lapply(unique(basindata$RI), function(y) doresample(basindata,y, max(table(basindata$RI)))))
  write.csv(train_resample, file = paste0("IMERG/Model_Data/", b, "_", 'train_resample.csv'))
}

data_24 <- do.call(rbind, lapply(unique(data_24$RI), function(y) doresample(data_24, y, max(table(data_24$RI)))))

write.csv(data_24, file = "IMERG/Model_Data/RI_Data_Global.csv")