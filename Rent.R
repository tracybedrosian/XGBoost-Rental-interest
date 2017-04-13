# Load packages and get data
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

dataTrain <- fromJSON("train.json")
vars <- setdiff(names(dataTrain), c("photos", "features"))
dataTrain <- map_at(dataTrain, vars, unlist) %>% tibble::as_tibble(.)

dataTest <- fromJSON("test.json")
vars <- setdiff(names(dataTest), c("photos", "features"))
dataTest <- map_at(dataTest, vars, unlist) %>% tibble::as_tibble(.)

dataTest$interest_level <- NA

dataTrain$dataset <- "train"
dataTest$dataset <- "test"
data <- rbind(dataTrain, dataTest)

# Extract number of photos
data$numphotos <- lengths(data$photos)
library(ggplot2)
hhigh <- hist(data$numphotos[data$interest_level == "high"], breaks = 100, plot=FALSE)
hhigh$counts=hhigh$counts/sum(hhigh$counts)
plot(hhigh)

hmed <- hist(data$numphotos[data$interest_level == "medium"], breaks = 100, plot=FALSE)
hmed$counts=hmed$counts/sum(hmed$counts)
plot(hmed)

hlow <- hist(data$numphotos[data$interest_level == "low"], breaks = 100, plot=FALSE)
hlow$counts=hlow$counts/sum(hlow$counts)
plot(hlow, xlim = c(0,25))

# Feature engineering manager skill
library(reshape2)
managerData <- data[, c(9, 11, 15)]
managerSpread <- dcast(managerData, formula = manager_id ~ interest_level, fun.aggregate = length)
managerSpread$manager_skill <- (managerSpread$high / (managerSpread$high + managerSpread$low + managerSpread$medium))
hist(managerSpread$manager_skill)
managerSkill <- managerSpread[, c(1, 6)]
data <- merge(data, managerSkill, by.x = "manager_id", all = TRUE)

# Feature engineering building popularity
library(reshape2)
data$building_id <- replace(data$building_id, data$building_id == 0, NA)
buildingData <- data[, c(4, 15)]
buildingSpread <- dcast(buildingData, formula = building_id ~ interest_level, fun.aggregate = length)
buildingSpread$building_pop <- (buildingSpread$high / (buildingSpread$high + buildingSpread$low + buildingSpread$medium))
hist(buildingSpread$building_pop)
buildingPop <- buildingSpread[, c(1, 6)]
data <- merge(data, buildingPop, by.x = "building_id", all = TRUE)

# Find correct address coordinates
library(ggmap)
address <- data[data$longitude == 0 | data$latitude == 0, ]$street_address
address
ny_address <- paste(address, ", new york")
address <- data.frame("street_address" = address)
coords <- sapply(ny_address, function(x) geocode(x, source = "google")) %>%
t %>%
data.frame %>%
cbind(address, .)
rownames(coords) <- 1:nrow(coords)
coords

data[data$longitude == 0,]$longitude <- coords$lon
data[data$latitude == 0,]$latitude <- coords$lat

# Visualize interest level by map location
library(ggplot2)
library(ggmap)

dataTrain <- subset(data, dataset == "train")
map <- get_map(location = c(lon = -73.9539, lat = 40.7108), zoom = 12,
                      maptype = "satellite")

dataTrain$longitude <- as.numeric(dataTrain$longitude)
dataTrain$latitude <- as.numeric(dataTrain$latitude)

ggmap(map) + geom_point(data = dataTrain, aes(x = longitude, y = latitude, fill = interest_level, alpha = 0.8), size = 1, shape = 21) + xlab("longitude") + ylab("latitude")

# Visualize interest level by price and bedrooms
ggplot(dataTrain, aes(x = bedrooms, y = price)) + geom_point(aes(colour = factor(interest_level))) + ylim(0, 15000)

# Visualize interest level by time of year
library(tidyr)
data <- separate(data = data, col = created, into = c("year", "month", "day_time"), sep = "-")
ggplot(data, aes(x = month, fill = interest_level)) + geom_bar(position = "dodge")

# Extract length of rental description
data$descriptors <- sapply(gregexpr("\\W+", data$description), length) + 1
ggplot(data, aes(x = interest_level, y = descriptors)) + geom_point() + geom_violin() + ylim(0,1000)

# Extract features
library(tidyr)
featuresData <- data[,c(10, 12, 17, 18)]

featuresExtracted <- featuresData %>%
  filter(map(features, is_empty) != TRUE) %>%
  tidyr::unnest(features)

featuresExtracted$features <- tolower(featuresExtracted$features)

featuresExtracted$interest_level <- as.factor(featuresExtracted$interest_level)
featuresExtracted$features <- as.factor(featuresExtracted$features)

# Identify common features and convert rare ones to Other
sort(table(featuresExtracted$features), decreasing = TRUE)[1:31]
library(forcats)
featuresExtracted$features <- fct_lump(featuresExtracted$features, 31)

# Reshape data to wide format
library(reshape2)
featuresSpread <- unique(featuresExtracted)
featuresSpread <- dcast(featuresSpread, formula = listing_id ~ features, fun.aggregate = length)

interest <- data[,c(12, 17, 18)]
interest <- unique(interest)
features <- merge(interest, featuresSpread, by.x = "listing_id", all = TRUE)

names(features) <- gsub(" ", "_", names(features))
features[, c(4:35)] <- lapply(features[, c(4:35)], factor)

# Get other features into wide form
otherFeatures <- data[, c(3, 4, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22)]
allFeatures <- merge(features, otherFeatures, by.x = "listing_id", by.y = "listing_id")
allFeatures <- allFeatures[, -c(42, 41)]

colnames(allFeatures)[16] <- "garden_patio"
colnames(allFeatures)[28] <- "prewar"

# XGBoost
library(xgboost)
library(Matrix)
library(caret)
library(Ckmeans.1d.dp)

train <- subset(allFeatures, interest_level.x == "high" | interest_level.x == "medium" | interest_level.x == "low")
test <- subset(allFeatures, is.na(interest_level.x))
train <- train[,-c(3,27)]
test <- test[,-c(3,27)]

train$interest_level.x<-as.integer(factor(train$interest_level.x))
y <- train$interest_level.x
y = y-1
train$interest_level.x = NULL
test$interest_level.x = NULL

seed <- 123

xgb_params = list(
  colsample_bytree = 1,
  subsample = 0.7,
  eta = 0.1,
  objective = 'multi:softprob',
  max_depth = 4,
  min_child_weight = 1,
  eval_metric = "mlogloss",
  num_class = 3,
  seed = seed
)

dtest <- xgb.DMatrix(data.matrix(test))

kfolds <- 7
folds <- createFolds(y, k = kfolds, list = TRUE, returnTrain = FALSE)
fold <- as.numeric(unlist(folds[1]))

x_train <- train[-fold,]
x_val <- train[fold,]

y_train <- y[-fold]
y_val <- y[fold]

dtrain = xgb.DMatrix(data.matrix(x_train), label = y_train)
dval = xgb.DMatrix(data.matrix(x_val), label = y_val)

gbdt = xgb.train(params = xgb_params,
                 data = dtrain,
                 nrounds = 475,
                 watchlist = list(train = dtrain, val=dval),
                 print_every_n = 100,
                 early_stopping_rounds = 50)

allpredictions =  (as.data.frame(matrix(predict(gbdt,dtest), nrow=dim(test), byrow=TRUE)))

allpredictions = cbind (allpredictions, test$listing_id)
names(allpredictions)<-c("high","low","medium","listing_id")
allpredictions = allpredictions[,c(1,3,2,4)]
write.csv(allpredictions,paste0(Sys.Date(),"-BaseModel-20Fold-Seed",seed,".csv"),row.names = FALSE)

imp <- xgb.importance(names(train),model = gbdt)
xgb.ggplot.importance(imp)
