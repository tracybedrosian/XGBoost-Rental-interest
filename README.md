# XGBoost-Rental-interest
Predicting interest in rentals using machine learning

Data obtained from Kaggle's Two Sigma Connect: Rental Listing Inquiries competition
Available from this link: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data

The goal of this Kaggle competition is to predict whether a NYC rental listing has high, medium, or low interest using various features of the listing such as price, bedrooms, location, and description. Here I explore the data in R to determine what features seem most important and contruct a predictive model using XGBoost that can calculate the probability that a listing is high, medium, or low interest.

## Load data
For the competition, data are supplied as JSON files divided into training and test sets. We will pull them into R and combine them into one dataset to be manipulated as a whole.

```
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
```
## Extract number of photos
Each rental listing is associated with a separate photos file. For now we'll ignore the photos themselves but extract the meta-data regarding the number of photos per listing. 

```
data$numphotos <- lengths(data$photos)
library(ggplot2)
hhigh <- hist(data$numphotos[data$interest_level == "high"], breaks = 100, plot=FALSE)
hhigh$counts=hhigh$counts/sum(hhigh$counts)
plot(hhigh, main = "High Interest", xlab = "Number of photos")

hmed <- hist(data$numphotos[data$interest_level == "medium"], breaks = 100, plot=FALSE)
hmed$counts=hmed$counts/sum(hmed$counts)
plot(hmed, main = "Medium Interest", xlab = "Number of photos")

hlow <- hist(data$numphotos[data$interest_level == "low"], breaks = 100, plot=FALSE)
hlow$counts=hlow$counts/sum(hlow$counts)
plot(hlow, xlim = c(0,25), main = "Low Interest", xlab = "Number of photos")
```
insert 3 histograms

## Determine manager skill
Each rental is listed by a particular manager so we'll check whether some managers are more skilled than others and create a feature for the proportion of high interest listings they have.

```
library(reshape2)
managerData <- data[, c(9, 11, 15)]
managerSpread <- dcast(managerData, formula = manager_id ~ interest_level, fun.aggregate = length)
managerSpread$manager_skill <- (managerSpread$high / (managerSpread$high + managerSpread$low + managerSpread$medium))
hist(managerSpread$manager_skill, main = "Manager Skill", xlab = "Skill")
managerSkill <- managerSpread[, c(1, 6)]
data <- merge(data, managerSkill, by.x = "manager_id", all = TRUE)
```
insert 1 histogram

## Determine building popularity
These rentals are located in NYC so many of them are located within the same building. We'll create a feature for the popularity of the building.

```
library(reshape2)
data$building_id <- replace(data$building_id, data$building_id == 0, NA)
buildingData <- data[, c(4, 15)]
buildingSpread <- dcast(buildingData, formula = building_id ~ interest_level, fun.aggregate = length)
buildingSpread$building_pop <- (buildingSpread$high / (buildingSpread$high + buildingSpread$low + buildingSpread$medium))
hist(buildingSpread$building_pop, main = "Building Popularity", xlab = "Popularity")
buildingPop <- buildingSpread[, c(1, 6)]
data <- merge(data, buildingPop, by.x = "building_id", all = TRUE)
```
insert 1 histogram

## Find correct address coordinates
Some rental listings lack complete address information and have been mis-assigned geographic coordinates outside of NY. We'll find and fix these listings by connecting to Google Maps API.

```
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
```

## "Location, location, location!" Visualize interest level on a map of NYC

```
library(ggplot2)
library(ggmap)

dataTrain <- subset(data, dataset == "train")
map <- get_map(location = c(lon = -73.9539, lat = 40.7108), zoom = 12,
                      maptype = "satellite")

dataTrain$longitude <- as.numeric(dataTrain$longitude)
dataTrain$latitude <- as.numeric(dataTrain$latitude)

ggmap(map) + geom_point(data = dataTrain, aes(x = longitude, y = latitude, fill = interest_level, alpha = 0.8), size = 1, shape = 21) + xlab("longitude") + ylab("latitude")
```
insert map

High interest rentals seem to be distributed all over NYC and not restricted to one specific neighborhood or even Manhatten. Other factors must contribute to interest level. 

## Visualize interest level by price and bedrooms

```
ggplot(dataTrain, aes(x = bedrooms, y = price)) + geom_point(aes(colour = factor(interest_level))) + ylim(0, 15000)
```
insert graph

Lower price rentals seem to be more popular across all apartment sizes.

## Visualize interest level by time of year
We have data for rental listings posted over a 3-month period which is fairly limited but we'll still check if interest level peaks during any of the months represented.

```
library(tidyr)
data <- separate(data = data, col = created, into = c("year", "month", "day_time"), sep = "-")
ggplot(data, aes(x = month, fill = interest_level)) + geom_bar(position = "dodge")
```
insert 1 plot

## Extract length of rental description
Each rental listing has an unstructured description. For now we'll just extract the length of that description as a new feature.

```
data$descriptors <- sapply(gregexpr("\\W+", data$description), length) + 1
ggplot(data, aes(x = interest_level, y = descriptors)) + geom_point() + geom_violin() + ylim(0,1000)
```

insert plot

Listings without a description seem to fall into the low interest category.

## Extract features
Each listing is associated with a nested list of tags or features. We'll extract those features and get them into a useable format.

```
library(tidyr)
featuresData <- data[,c(10, 12, 17, 18)]

featuresExtracted <- featuresData %>%
  filter(map(features, is_empty) != TRUE) %>%
  tidyr::unnest(features)

featuresExtracted$features <- tolower(featuresExtracted$features)

featuresExtracted$interest_level <- as.factor(featuresExtracted$interest_level)
featuresExtracted$features <- as.factor(featuresExtracted$features)
```

There are many, many features so we'll just find the most common ones and lump the rare ones into an "other" category.

```
sort(table(featuresExtracted$features), decreasing = TRUE)[1:31]
library(forcats)
featuresExtracted$features <- fct_lump(featuresExtracted$features, 31)
```

## Reshape features into wide format
We'll get these data into a structure that works with our other features for eventual modeling.

```
library(reshape2)
featuresSpread <- unique(featuresExtracted)
featuresSpread <- dcast(featuresSpread, formula = listing_id ~ features, fun.aggregate = length)

interest <- data[,c(12, 17, 18)]
interest <- unique(interest)
features <- merge(interest, featuresSpread, by.x = "listing_id", all = TRUE)

names(features) <- gsub(" ", "_", names(features))
features[, c(4:35)] <- lapply(features[, c(4:35)], factor)
```

And we'll collect the other features of interest and merge everything into one data frame.

```
otherFeatures <- data[, c(3, 4, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22)]
allFeatures <- merge(features, otherFeatures, by.x = "listing_id", by.y = "listing_id")
allFeatures <- allFeatures[, -c(42, 41)]

colnames(allFeatures)[16] <- "garden_patio"
colnames(allFeatures)[28] <- "prewar"
```

## Training a model and making predictions
Finally we'll use XGBoost to train a model to predict the interest level of a particular rental listing based on its features.

```
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
write.csv(allpredictions,paste0(Sys.Date(),"-XGBoost",seed,".csv"),row.names = FALSE)

imp <- xgb.importance(names(train),model = gbdt)
xgb.ggplot.importance(imp)
```

insert imp plot

The most important features are depicted. 

