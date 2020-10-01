

library('xgboost')
library(caret)
library(MASS)
library(leaps)
library(car)
library(LSTS)
library(lmtest)
library(randomForest)
library(dummies)
library(dplyr)
library(gam)
library(RRF)


dat<-read.csv('bnbdata_final.csv')

# factorizing variables
dat$room_price = as.numeric(dat$room_price)
dat$wifi = factor(dat$wifi)
dat$washing_machine = factor(dat$washing_machine)
dat$iron = factor(dat$iron)
dat$laptop_workspace = factor(dat$laptop_workspace)
dat$TV = factor(dat$TV)
dat$cable_TV = factor(dat$cable_TV)
dat$hot_water = factor(dat$hot_water)
dat$heating = factor(dat$heating)
dat$airconditioning = factor(dat$airconditioning)
dat$microwave = factor(dat$microwave)
dat$kitchen = factor(dat$kitchen)
dat$hanger = factor(dat$hanger)
dat$paid_parking = factor(dat$paid_parking)
dat$free_parking = factor(dat$free_parking)
dat$hair_dryer = factor(dat$hair_dryer)
dat$first_aid_kit = factor(dat$first_aid_kit)
dat$essentials = factor(dat$essentials)
dat$shampoo = factor(dat$shampoo)
dat$breakfast = factor(dat$breakfast)
dat$tableware = factor(dat$tableware)
dat$stove = factor(dat$stove)
dat$luggage_store = factor(dat$luggage_store)
dat$long_stays_allowed = factor(dat$long_stays_allowed)
dat$greetings_from_host = factor(dat$greetings_from_host)
dat$extra_bedclothes = factor(dat$extra_bedclothes)
dat$pocket_wifi = factor(dat$pocket_wifi)
dat$coffee_maker = factor(dat$coffee_maker)
dat$balcony = factor(dat$balcony)
dat$garden_or_backyard = factor(dat$garden_or_backyard)
dat$river_lake_side = factor(dat$river_lake_side)
dat$oven = factor(dat$oven)
dat$refrigerator = factor(dat$refrigerator)
dat$doorlock = factor(dat$doorlock)
dat$guest_entrance = factor(dat$guest_entrance)
dat$ethernet = factor(dat$ethernet)
dat$dish_washer = factor(dat$dish_washer)
dat$plug_cover = factor(dat$plug_cover)

dat$SECURITY_ITEMS = factor(dat$SECURITY_ITEMS)
dat$BABY_ITEMS = factor(dat$BABY_ITEMS)
dat$FIRE_ITEMS = factor(dat$FIRE_ITEMS)
dat$LUXURY_ITEMS = factor(dat$LUXURY_ITEMS)


#각 type별 이상치 제거
dat$idx=1:nrow(dat)
boxplot(room_price~type_airbnb,data=dat, col = 'skyblue',horizontal=TRUE)
L<-tapply(dat$room_price,dat$type_airbnb,boxplot.stats)
s=1:22
t=vector('character',length(s))
u=NULL
for(i in 1:22){
  s[i]=L[[i]][4]
  t[i]=names(L[i])
  u[i]=length(s[[i]])
}

names(s)<-t
v=NULL
for(i in 1:22){
  v=c(v,rep(t[i],u[i]))
}
s=data.frame(cbind(v,unlist(s,use.names = F)),stringsAsFactors = F)
s$V2=as.numeric(s$V2)
colnames(s)= c('type_airbnb','room_price')

p=merge(s,dat,by=c('type_airbnb','room_price'))
p=distinct(p)
data=dat[-p$idx,]
data=data[,-50]
dat=dat[,-50]

#변수 살펴보기
tapply(dat$room_price,dat$type_airbnb,mean)
tapply(dat$room_price,dat$type_airbnb,length)
par(las=1)
boxplot(room_price~type_airbnb,data=data, col = 'skyblue',horizontal=TRUE)

tapply(dat$room_price,dat$district,mean)
sort(tapply(dat$room_price,dat$district,length))
par(las=1)
boxplot(room_price~district,data=data, col = 'skyblue',horizontal=TRUE)

hist(data$room_price,breaks = 50)
hist(data$room_price[data$room_price<150000],breaks = 40)
data1=data[data$room_price<150000,]

# # 앞뒤 5% 제거 
# or<-order(dat$room_price)
# idx<-or[c(1:98,1866:1963)]
# dat<-dat[-idx,]



#train/test data
set.seed(828)
i<-createDataPartition(y=data1$room_price,p=0.8,list = F)
traindata<-data1[i,]
testdata<-data1[-i,]
train.label<-traindata[,1]
test.label<-testdata[,1]
nrow(traindata)
nrow(testdata)
head(traindata)



######## Randomforest ##########varImpPlot(bnbRFTrain) # check important variables

bnbRFTrain = randomForest(room_price ~ ., data = traindata, ntree=300, mtry = 17)
par(mfrow=c(1,1))
plot(bnbRFTrain) # choose best tree numbers

bnbRF_pred=predict(bnbRFTrain,newdata=testdata) # predict
RandomForest_RMSE<-sqrt(mean((testdata$room_price - bnbRF_pred)^2)) 
RandomForest_RMSE
var(bnbRF_pred)/var(testdata$room_price)
sqrt(mean((log(testdata$room_price) - log(bnbRF_pred))^2)) 

########### GRF ##############
x <- data1[,-1]
y <- data1[,1]
x.train <- x[i,]; y.train <- y[i]
x.test <- x[-i,]; y.test <- y[-i]

rf <- RRF(x.train,flagReg = 0,y.train,ntree=300, mtry = 17) # original rf
imp <- rf$importance
impRF <- imp/max(imp) #normalize

#build a GRF with gamma = 1(penalty term)
gamma = 1.002
coefReg <- (1-gamma) + gamma*impRF
GRF <- RRF(x.train,y.train,flagReg=1,coefReg=coefReg,ntree=300, mtry = 17)
length(GRF$feaSet)
GRF_RF <- RRF(x.train[,GRF$feaSet],flagReg=1,y.train)
pred <- predict(GRF_RF,x.test[,GRF$feaSet])
GRF_RMSE<-sqrt(mean((pred-y.test)^2))
GRF_RMSE
var(pred)/var(testdata$room_price)
pred <- predict(rf,x.test)
(mean((pred-y.test)^2))

gamma = 1.004
coefReg <- (1-gamma) + gamma*impRF
GRF <- RRF(x.train,y.train,flagReg=1,coefReg=coefReg,ntree=300, mtry = 17)
length(GRF$feaSet)
GRF_RF <- RRF(x.train[,GRF$feaSet],flagReg=1,y.train)
pred <- predict(GRF_RF,x.test[,GRF$feaSet])
GRF_RMSE<-sqrt(mean((pred-y.test)^2))
GRF_RMSE
var(pred)/var(testdata$room_price)
pred <- predict(rf,x.test)
(mean((pred-y.test)^2))

newdata<-read.csv('newdata.csv')
predict(bnbRFTrain, newdata, type="response",
        norm.votes=TRUE, predict.all=FALSE, proximity=FALSE, nodes=FALSE,
        cutoff, ...)
######## GAM ##########

gam.fit1<-gam(room_price ~ district + s(number_of_bedrooms) + s(number_of_beds) + 
                s(number_of_individual_bathrooms) + s(number_of_shared_bathrooms) + 
                refund_rule + type_airbnb + wifi + essentials + laptop_workspace + 
                airconditioning + heating + TV + cable_TV + washing_machine + 
                kitchen + hanger + paid_parking + free_parking + breakfast + 
                shampoo + coffee_maker + tableware + oven + microwave + refrigerator + 
                stove + luggage_store + long_stays_allowed + hair_dryer + 
                first_aid_kit + LUXURY_ITEMS + FIRE_ITEMS + iron + hot_water + 
                doorlock + greetings_from_host + extra_bedclothes + garden_or_backyard + 
                guest_entrance + SECURITY_ITEMS + BABY_ITEMS + ethernet + 
                balcony + dish_washer + pocket_wifi + plug_cover + river_lake_side, data = traindata)
summary(gam.fit1)
par(mfrow=c(3,3))
plot(gam.fit1)


yhat.gam1<- predict(gam.fit1,newdata=testdata)
GAM_RMSE<-sqrt(mean((yhat.gam1-testdata$room_price)^2))
GAM_RMSE
dim(traindata)
dim(testdata)
##########XGboost#############


# dummy

dat.n<-dummy.data.frame(data1)

#train/test data
set.seed(828)
traindata.n<-dat.n[i,]
testdata.n<-dat.n[-i,]
train.label<-traindata.n[,1]
test.label<-testdata.n[,1]

train = xgb.DMatrix(as.matrix(traindata.n[,-1]),label=train.label)
test = xgb.DMatrix(as.matrix(testdata.n[,-1]),label=test.label)


#XGboost
watchlist = list(train = train, test = test)
param = list(booster = 'gbtree',
             lambda = 1,
             gamma=0,
             subsample = 0.5,
             colsample_bytree =0.5,
             eta = 0.05,   
             max_depth = 10,
             max_leaf_nodes=1024,
             eval_metric = "rmse",
             objective = 'reg:linear',
             seed=828)

XGtr = xgb.train(params = param, data = train,
                 nrounds = 100, watchlist=watchlist)

tp<-predict(XGtr,as.matrix(testdata.n)[,-1],type='class')
XGBoost_RMSE<-sqrt(mean((testdata.n$room_price - tp )^2)) 
XGBoost_RMSE
postResample(tp, testdata.n$room_price)


var(tp)/var(testdata$room_price)

######## Linear Regression ##########
fit1 <- lm(log(room_price) ~ ., data=data1)
summary(fit1)
par(mfrow=c(2,2)); plot(fit1)

# Remove outliers
data1<-data1[-c(333,1429),]
fit2 <-lm(log(room_price) ~ ., data=data1)
plot(fit2)

dat.r<-fit2$model

# subset selection
null=lm(`log(room_price)`~1,data=dat.r)
full=lm(`log(room_price)`~.,data=dat.r)
step(full, scope=list(lower=null, upper=full), direction = "both")

fit3<-lm(formula = `log(room_price)` ~ district + number_of_beds + 
           number_of_shared_bathrooms + refund_rule + type_airbnb + 
           airconditioning + TV + washing_machine + kitchen + paid_parking + 
           free_parking + breakfast + shampoo + tableware + refrigerator + 
           stove + long_stays_allowed + LUXURY_ITEMS + iron + hot_water + 
           garden_or_backyard + guest_entrance + balcony + dish_washer + 
           river_lake_side, data = dat.r)




summary(fit3)

hist(fit3$residuals,breaks=50,col="lavender")

# 3 assumptions
shapiro.test(fit3$residuals)
hist(fit3$residuals)
ncvTest(fit3)



dwtest(fit3)
Box.Ljung.Test(fit3$residuals)
vif(fit3)

dat.r<-fit3$model

traindata<-dat.r[i,]
testdata<-dat.r[-i,]


fit.train<-lm(formula = `log(room_price)` ~ district + number_of_beds + 
                number_of_shared_bathrooms + refund_rule + type_airbnb + 
                airconditioning + TV + washing_machine + kitchen + paid_parking + 
                free_parking + breakfast + shampoo + tableware + refrigerator + 
                stove + long_stays_allowed + LUXURY_ITEMS + iron + hot_water + 
                garden_or_backyard + guest_entrance + balcony + dish_washer + 
                river_lake_side, data = traindata)

# rmse

pre<-predict(fit.train,testdata)
mse<-mean((testdata$`log(room_price)`-pre)^2) 
rmse<-sqrt(mse) 
Regression_RMSE<-sqrt(mean((exp(testdata$`log(room_price)`) - exp(pre))^2)) 
Regression_RMSE


# Comparison
data.frame(Regression_RMSE,RandomForest_RMSE,GRF_RMSE,XGBoost_RMSE, GAM_RMSE)


########### 최종모델 : GRF  ##############

newdata<-read.csv('newdata.csv')


# factorizing variables
newdata$room_price = as.numeric(newdata$room_price)
newdata$wifi = factor(newdata$wifi)
newdata$washing_machine = factor(newdata$washing_machine)
newdata$iron = factor(newdata$iron)
newdata$laptop_workspace = factor(newdata$laptop_workspace)
newdata$TV = factor(newdata$TV)
newdata$cable_TV = factor(newdata$cable_TV)
newdata$hot_water = factor(newdata$hot_water)
newdata$heating = factor(newdata$heating)
newdata$airconditioning = factor(newdata$airconditioning)
newdata$microwave = factor(newdata$microwave)
newdata$kitchen = factor(newdata$kitchen)
newdata$hanger = factor(newdata$hanger)
newdata$paid_parking = factor(newdata$paid_parking)
newdata$free_parking = factor(newdata$free_parking)
newdata$hair_dryer = factor(newdata$hair_dryer)
newdata$first_aid_kit = factor(newdata$first_aid_kit)
newdata$essentials = factor(newdata$essentials)
newdata$shampoo = factor(newdata$shampoo)
newdata$breakfast = factor(newdata$breakfast)
newdata$tableware = factor(newdata$tableware)
newdata$stove = factor(newdata$stove)
newdata$luggage_store = factor(newdata$luggage_store)
newdata$long_stays_allowed = factor(newdata$long_stays_allowed)
newdata$greetings_from_host = factor(newdata$greetings_from_host)
newdata$extra_bedclothes = factor(newdata$extra_bedclothes)
newdata$pocket_wifi = factor(newdata$pocket_wifi)
newdata$coffee_maker = factor(newdata$coffee_maker)
newdata$balcony = factor(newdata$balcony)
newdata$garden_or_backyard = factor(newdata$garden_or_backyard)
newdata$river_lake_side = factor(newdata$river_lake_side)
newdata$oven = factor(newdata$oven)
newdata$refrigerator = factor(newdata$refrigerator)
newdata$doorlock = factor(newdata$doorlock)
newdata$guest_entrance = factor(newdata$guest_entrance)
newdata$ethernet = factor(newdata$ethernet)
newdata$dish_washer = factor(newdata$dish_washer)
newdata$plug_cover = factor(newdata$plug_cover)

newdata$SECURITY_ITEMS = factor(newdata$SECURITY_ITEMS)
newdata$BABY_ITEMS = factor(newdata$BABY_ITEMS)
newdata$FIRE_ITEMS = factor(newdata$FIRE_ITEMS)
newdata$LUXURY_ITEMS = factor(newdata$LUXURY_ITEMS)


# 원래 predict 식 : predict(GRF_RF,x.test[,GRF$feaSet]) 
# x.test[,GRF$feaSet] 와 newdata[,GRF$feaSet]의 type을 맞추기 위해 데이터 rbind -> newdata2 생성
newdata2<-rbind(x.test[,GRF$feaSet],newdata[,GRF$feaSet])

# nrow(newdata2) : 370 : 우리가 알고싶은 데이터의 위치
predict(GRF_RF,newdata2[370,])

# 80370.52 원


# 회귀식 요인분석 변화가 실제 가격 예측값 변화와 유사한지 확인해보겠습니다. =
# breakfast를 0 -> 1 로 바꿀거에요.

newdata$breakfast<-as.numeric(newdata$breakfast)
newdata$breakfast<-1
newdata$breakfast<-as.factor(newdata$breakfast)
newdata2<-rbind(x.test[,GRF$feaSet],newdata[,GRF$feaSet])
predict(GRF_RF,newdata2[370,])
# pred 82598.38 

#회귀분석 모델로 예측한 가격 변화 : 80370.52*exp(0.0317) = 82942.38 원


# 요인분석이 괜찮게 되었음을 확인 ㄴㅅㄱ(ㅇ)


newdata$dish_washer<-as.numeric(newdata$dish_washer)
newdata$dish_washer<-1
newdata$dish_washer<-as.factor(newdata$dish_washer)
newdata2<-rbind(x.test[,GRF$feaSet],newdata[,GRF$feaSet])
predict(GRF_RF,newdata2[370,])
# 81433.95


######

newdata2<-data1[144,-1]
newdata2<-rbind(x.test[,GRF$feaSet],newdata2[,GRF$feaSet])

predict(GRF_RF,newdata2[370,])
# 24641.06
#실제가격 22424
#최저요금   22030
#최고요금   98701
#기본요금   32900
#주말가격   37835


newdata3<-data1[214,-1]
newdata3<-rbind(x.test[,GRF$feaSet],newdata3[,GRF$feaSet])
predict(GRF_RF,newdata3[370,])
#23681.19 
#실제가격 23485
#최저요금   12725
#최고요금   54535
#기본요금   18178
#주말가격   20905


newdata4<-data1[828,-1]
newdata4<-rbind(x.test[,GRF$feaSet],newdata4[,GRF$feaSet])
predict(GRF_RF,newdata2[370,])
#77976.25
#실제가격 84406
#최저가격 32178
#최고가격 137906
#적정각격  45969
#주말가격  52864


newdata5<-data1[1322,-1]
newdata5<-rbind(x.test[,GRF$feaSet],newdata5[,GRF$feaSet])
predict(GRF_RF,newdata5[370,])
#54086.85
#실제가격65115
#최저가격    26,386 
#최고가격    113,084 
#적정가격    37,695 
#주말가격    43,349 

newdata6<-data1[140,-1]
newdata6<-rbind(x.test[,GRF$feaSet],newdata6[,GRF$feaSet])
predict(GRF_RF,newdata6[370,])
#24523.04
#실제각격 21356
#최저가격    26872
#최고가격    115165 
#적정가격    38388
#주말가격     44146



newdata7<-data1[253,-1]
newdata7<-rbind(x.test[,GRF$feaSet],newdata7[,GRF$feaSet])
predict(GRF_RF,newdata7[370,])
#29444.22
#실제각격 23485
#최저가격    23552
#최고가격    100936 
#적정가격    33645
#주말가격    38692


