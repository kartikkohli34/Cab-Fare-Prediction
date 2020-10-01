rm(list=ls())
setwd("C:\\Users\\ASUS\\Downloads\\Compressed")
getwd()

# loading all required librares.
install.packages (c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
                    "MASS", "rpart", "gbm", "ROSE", "sampling", "DataCombine", "inTrees", "xgboost"))

# loading the data
train=read.csv("train_cab.csv",header=TRUE)
test=read.csv("test.csv",header=TRUE)
x=unique.data.frame(train)
train[which(train$pickup_datetime==43 ),]
train = train[-which(train$fare_amount==43 ),]
# exploring the data.

str(train)
str(test)
summary(train)
summary(test)
head(train,5) 
head(train,5)
dim(train)

# converting the features in the required data types.
train$fare_amount = as.numeric(as.character(train$fare_amount))
train$passenger_count=round(train$passenger_count) # converting into integer values

class(train$fare_amount)

########### data cleaning  #############
# fare amount cannot be less than one 
# considring fare amount 453 as max and removing all the fare amount greater than 453, as chances are
# very less of fare amount having 4000 and 5000 ...etc
train[which(train$fare_amount < 1 ),]

nrow(train[which(train$fare_amount < 1 ),]) # to show the count i.e.,5

train = train[-which(train$fare_amount < 1 ),]  # removing those values.

train[which(train$fare_amount>453),]

nrow(train[which(train$fare_amount >453 ),]) # to show the count i.e., 2

train = train[-which(train$fare_amount >453 ),]  # removing those values.

# passenger count cannot be Zero
# even if we consider suv max seat is 6, so removing passenger count greater than 6.
train[which(train$passenger_count < 1 ),]
nrow(train[which(train$passenger_count < 1 ),]) # to show count, that is 58
train=train[-which(train$passenger_count < 1 ),] # removing the values
train[which(train$passenger_count >6 ),]
nrow(train[which(train$passenger_count >6 ),]) # to show count, that is 20
train=train[-which(train$passenger_count >6 ),] # removing the values
# Latitudes range from -90 to 90.Longitudes range from -180 to 180.
# Removing which does not satisfy these ranges.

nrow(train[which(train$pickup_longitude >180 ),])
nrow(train[which(train$pickup_longitude < -180 ),])
nrow(train[which(train$pickup_latitude > 90 ),])
nrow(train[which(train$pickup_latitude < -90 ),])
nrow(train[which(train$dropoff_longitude > 180 ),])
nrow(train[which(train$dropoff_longitude < -180 ),])
nrow(train[which(train$dropoff_latitude < -90 ),])
nrow(train[which(train$dropoff_latitude > 90 ),])
train = train[-which(train$pickup_latitude > 90),] # removing one data point


# Also we will see if there are any values equal to 0.
nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])

# removing those data points.
train=train[-which(train$pickup_longitude == 0 ),]
train=train[-which(train$dropoff_longitude == 0),]


# checking for missing values.
sum(is.na(train))
sum(is.na(test))
train=na.omit(train) # we have removed the missing values...as they are less,,..likely 50 to 60 missing values.
sum(is.na(train))  



# new features will be year,month,day_of_week,hour
# Convert pickup_datetime from factor to date time
train$pickup_datetime=as.Date(train$pickup_datetime)
pickup_time = strptime(train$pickup_datetime,format='%Y-%m-%d %H:%M:%S UTC')
train$date = as.integer(format(train$pickup_date,"%d"))# Monday = 1
train$month = as.integer(format(train$pickup_date,"%m"))
train$year = as.integer(format(train$pickup_date,"%Y"))


# for test data set.
test$pickup_datetime=as.Date(test$pickup_datetime)
pickup_time = strptime(test$pickup_datetime,format='%Y-%m-%d %H:%M:%S UTC')
test$date = as.integer(format(test$pickup_date,"%d"))# Monday = 1
test$month = as.integer(format(test$pickup_date,"%m"))
test$year = as.integer(format(test$pickup_date,"%Y"))

# outlier
library(ggplot2)
#pl1 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
#pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# deriving the new feature, distance from the given coordinates.
deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}
train$distance = haversine(train$pickup_longitude,train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude)
test$distance = haversine(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)


# removing the features, which were used to create new features.
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,pickup_datetime))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,pickup_datetime))

str(train)

summary(train)
nrow(train[which(train$distance ==0 ),])
nrow(test[which(test$distance==0 ),])
nrow(train[which(train$distance >130 ),]) # considering the distance 130 as max and considering rest as outlier.
nrow(test[which(test$distance >130 ),])

# removing the data points by considering the above conditions,
train=train[-which(train$distance ==0 ),]
train=train[-which(train$distance >130 ),]
test=test[-which(test$distance ==0 ),]

############################feature selection################################
numeric_index = sapply(train,is.numeric) #selecting only numeric
numeric_data = train[,numeric_index]
cnames = colnames(numeric_data)

#Correlation analysis for numeric variables
library(corrgram)
corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#removing date/ Dimension Reduction
train = subset(train,select=-date)
#remove from test set
test = subset(test,select=-date)


############################feature scaling#############################
install.packages("car")
library(car)
library(MASS)
qqnorm(train$fare_amount) # qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line 
truehist(train$fare_amount) # truehist() scales the counts to give an estimate of the probability density.

d=density(train$fare_amount)
plot(d,main="distribution")
polygon(d,col="green",border="red")

D=density(train$distance)
plot(D,main="distribution")
polygon(D,col="green",border="red")

A=density(test$distance)
plot(A,main="distribution")
polygon(A,col="black",border="red")

#Normalisation
# log transformation.
train$fare_amount=log1p(train$fare_amount)
test$distance=log1p(test$distance)
train$distance=log1p(train$distance)

# checking back features afte rNormal Distribution transformation.
d=density(train$fare_amount)
plot(d,main="distribution")
polygon(d,col="green",border="red")

D=density(train$distance)
plot(D,main="distribution")
polygon(D,col="red",border="black")

A=density(test$distance)
plot(A,main="distribution")
polygon(A,col="black",border="red")




## to make sure that we dont have any missing values
sum(is.na(train))
train=na.omit(train)


########################model building##############################
# preparing the data
set.seed(1200)
Train.index = sample(1:nrow(train), 0.9 * nrow(train))
Train = train[ Train.index,]
Test  = train[-Train.index,]
head(Test[,2:6],5)
TestData=test
library(glmnet)



###################multicollinearity check##################################
install.packages('usdm')
library(usdm)
vif(train[,-1])
vifcor(train[,-1], th = 0.9)

########################decision tree regressor##############################
library(rpart)
DT=rpart(fare_amount~.,data=Train,method="anova")
predictions_tree=predict(DT,Test[,2:5])
#predictions_test=predict(DT,TestData)                   #MAss for practice data
summary(DT)
DT

###########################linear regression##################################
# Run Linear regresion Model
linear_model=lm(fare_amount~.,data=Train)
summary(linear_model)

#Predict
predict_lm=predict(linear_model,Test[,2:5])


######################random forest regressor#####################################
library(randomForest)
random_model = randomForest(fare_amount~ ., Train, importance = TRUE, ntree = 500)


#Extract rules fromn random forest
#transform rf /object to an inTrees' format
library(inTrees)
treeList = RF2List(random_model)  

#Extract rules
rules= extractRules(treeList, Train[,2:5])

#Visualize some rules
rules[1:2,]
#Make rules more readable:
readrules = presentRules(rules, colnames(Train))
readrules[1:2,]

#Predict test data using random forest model
RF_Predictions = predict(random_model, Test[,2:5])
#RF_test=predict(random_model, TestData)


####################################XGBoost##########################################
### for xgboost it is required to make date variable as factor. trainset$Date <- as.factor(trainset$Date)
#Develop Model on training data

library(xgboost)
library(gbm)
fit_XGB = gbm(fare_amount~., data = Train, n.trees = 500, interaction.depth = 2)

#Lets predict for test data
pred_XGB_test = predict(fit_XGB, Test[,2:5], n.trees = 500)


## accuracy check
#defining the function (to find the error percentage)
mape=function(av,pv){
  mean(abs((av-pv)/av))*100 #av=actual value and pv= predicted value
}
library(DMwR)
##############linear regression model###################
mape(Test[,1],predict_lm)
# 7.78

regr.eval(Test[,1],predict_lm)
# mae        mse       rmse       mape 
#0.17510848 0.08089190 0.28441501.07778767 

################# decision tree ########################
mape(Test[,1],predictions_tree)
# 8.61

regr.eval(Test[,1],predictions_tree)
# mae        mse       rmse       mape 
#0.19292435 0.07946429 0.28189411 0.08617598 

###############random_forest################################
mape(Test[,1],RF_Predictions)
# 10.06

regr.eval(Test[,1],RF_Predictions)
# mae        mse       rmse       mape 
#0.22312437 0.09489477 0.30804995 0.10069728 

####################Xgboost####################################
### for xgboost it is required to make date variable as factor. 

mape(Test[,1],pred_XGB_test) 
#7.44

regr.eval(Test[,1],pred_XGB_test)
# mae        mse       rmse       mape 
#0.16544827 0.06464456 0.25425294 0.07449069 

#Predicting the fare amount for test data using Xgboost

predict_test= predict(fit_XGB ,TestData)

TestData$predicted_fare=predict_test
head(TestData)

###############Output of test data pricted values##################
write.csv(TestData, "predicted_testdata.csv", row.names = F)

