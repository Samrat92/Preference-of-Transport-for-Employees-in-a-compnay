library(ggplot2)
library(ggcorrplot)
library(ellipse)
library(RColorBrewer)
library(nFactors)
library(psych)
library(lattice)
library(caTools)
library(rpart)
library(rpart.plot)
library(rattle)
library(data.table)
library(ineq)
library(data.table)
library(StatMeasures)
library(htmlwidgets)
library(DataExplorer)
library(corrplot)
library(partykit)
library(dplyr)
library(purrr)
library(InformationValue)
library(car)
library(ROCR)
library(MASS)
library(class)
library(caret)
library(e1071)
library(Matrix)
library(DMwR)
library(ipred)
library(gbm)


setwd("C:/Users/Samrat/Documents/R/Directories/")
getwd()
data = read.csv("Cars_edited.csv")


head(data)

names(data)
str(data)
summary(data)
attach(data)


data[is.na(data)] = 0
sum(is.na(data))

hist(Age,density = 20, col = "blue")
hist(Work.Exp,density = 20, col = "green")
hist(Salary,density = 20, col = "dark red")
hist(Distance,density = 20, col = "light blue")


boxplot(Work.Exp, col = "light blue")
boxplot(Age, col = "cyan")
boxplot(Salary, col = "pink")
boxplot(Distance, col = "dark green")

IQRWorkexp = IQR(data$Work.Exp)
LLWorkexp = quantile(data$Work.Exp,0.25) - 1.5*IQRWorkexp
ULWorkexp = quantile(data$Work.Exp,0.75) + 1.5*IQRWorkexp
WorkexpOut = subset(data, Work.Exp < LLWorkexp | Work.Exp > ULWorkexp)
dim(WorkexpOut)

IQRAge = IQR(data$Age)
LLAge = quantile(data$Age,0.25) - 1.5*IQRAge
ULAge = quantile(data$Age,0.75) + 1.5*IQRAge
AgeOut = subset(data, Age < LLAge | Age > ULAge)
dim(AgeOut)


IQRSalary = IQR(data$Salary)
LLSalary = quantile(data$Salary,0.25) - 1.5*IQRSalary
ULSalary = quantile(data$Salary,0.75) + 1.5*IQRSalary
SalaryOut = subset(data, Salary < LLSalary | Salary > ULSalary)
dim(SalaryOut)


IQRDistance = IQR(data$Distance)
LLDistance = quantile(data$Distance,0.25) - 1.5*IQRDistance
ULDistance = quantile(data$Distance,0.75) + 1.5*IQRDistance
DistanceOut = subset(data, Distance < LLDistance | Distance > ULDistance)
dim(DistanceOut)


plot(Distance~Transport, col = c("yellow","red","orange"))
plot(Salary ~ Gender, col = c("green","blue"))

data$Transport = as.numeric(data$Transport)

data$Transport[data$Transport == 1] = 0


data$Transport[data$Transport == 3] = 0


data$Transport[data$Transport == 2] = 1




prop.table(table(data$Transport))
barplot(prop.table(table(data$Transport)), col = rainbow(3),main = "No Car vs Car")



data2 = data[,sapply(data, is.numeric)]

summary(data2)
str(data2)



corr.matrix = round(cor(data2),3) 
corr.matrix
plot_correlation(data)

ggcorrplot(corr.matrix, type = "lower", ggtheme = ggplot2::theme_gray,
           show.legend = TRUE, show.diag = TRUE, colors = c("cyan","white","sky blue"),
           lab = TRUE)

my_colors = brewer.pal(7, "Blues")
my_colors = colorRampPalette(my_colors)(100)
plotcorr(corr.matrix , col=my_colors[corr.matrix*50+50] , mar=c(1,1,1,1), )

cortest.bartlett(corr.matrix)

KMO(corr.matrix)

e = eigen(corr.matrix)
ev = e$values
ev


plot(ev, xlab = "Factors", ylab="Eigen Value", pch=20, col="blue")
lines(ev, col="red")
eFactors = fa(data2, nfactors=1, rotate="varimax", fm = "minres")
eFactors
fa.diagram(eFactors)

data.sub = data %>% select(-"Work.Exp",-"Salary",-"Age",-"license",-"Distance")
data3 = cbind(data.sub, sei = eFactors$scores)

plot_correlation(data3)

colnames(data3)[5] = "CustomerProfile"
str(data3)
summary(data3)


set.seed(10)
spl = sample.split(data3$Transport, SplitRatio = 0.75)
train = subset(data3, spl == T)
test = subset(data3, spl == F)

dim(train)
dim(test)
prop.table(table(train$Transport))
prop.table(table(test$Transport))


set.seed(1000)
summary(data3$Transport)
balanced.data = SMOTE(Transport ~.,perc.over = 100 , data3 , k = 5, perc.under = 812)
table(balanced.data$Transport)

set.seed(144)
split = sample.split(balanced.data$Transport, SplitRatio = 0.75)
train.bal = subset(balanced.data, split == T)
test.bal = subset(balanced.data, split == F)

table(train.bal$Transport)
table(test.bal$Transport)

LRmodel = glm(Transport ~., data = train, family = binomial)
summary(LRmodel)

predTest = predict(LRmodel, newdata = test, type = 'response')
cmLR = table(test$Transport, predTest>0.1)
cmLR
sum(diag(cmLR))/sum(cmLR)


ROCRpred = prediction(predTest, test$Transport)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf)



LR.smote = glm(Transport ~., data = train.bal, family = binomial)
summary(LR.smote)

pred.Test.smote = predict(LR.smote, newdata = test.bal, type = 'response')
cm.smote = table(test.bal$Transport, pred.Test.smote >0.1)
cm.smote
sum(diag(cm.smote))/sum(cm.smote)

ROCRpred2 = prediction(pred.Test.smote, test.bal$Transport)
as.numeric(performance(ROCRpred2, "auc")@y.values)
perf2 = performance(ROCRpred2, "tpr","fpr")
plot(perf2)



NBmodel = naiveBayes(Transport ~., data = train)
NBpredTest = predict(NBmodel, newdata = test, type = "class")
tabNB = table(test$Transport, NBpredTest)
tabNB


NBmodel.bal = naiveBayes(Transport ~., data = train.bal)
NBpredTest.bal = predict(NBmodel.bal, newdata = test.bal)
tabNB.bal = table(test.bal$Transport, NBpredTest.bal)
tabNB.bal


train.num = train[,sapply(train, is.numeric)]
test.num = test[,sapply(test, is.numeric)]
names(train.num)
KNNmodel = knn(train = train.num, test = test.num, cl = train$Transport, k = 3)
tabKNN = table(test$Transport, KNNmodel)
tabKNN 


train.num.bal = train.bal[,sapply(train.bal, is.numeric)]
test.num.bal = test.bal[,sapply(train.bal, is.numeric)]
names(train.num.bal)
KNNmodel.bal = knn(train = train.num.bal, test = test.num.bal, cl = train.bal$Transport, k = 6)
tabKNN.bal = table(test.bal$Transport, KNNmodel.bal)
tabKNN.bal


knn.fit = train(Transport ~., data = train.bal, method = "knn",
                trControl = trainControl(method = "cv", number = 3),
                tuneLength = 10)
knn.fit

predKNN.fit = predict(knn.fit, newdata = test.num.bal$Transport)
predKNN.fit


bagging = bagging(Transport ~.,data=train, control=rpart.control(maxdepth=5, minsplit=4))

test$pred.Transport = predict(bagging, test)
table(test$Transport,test$pred.Transport)

gbm.fit = gbm(
  formula = Transport ~ .,
  distribution = "bernoulli",
  data = train.bal,
  n.trees = 5000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, 
  verbose = FALSE
)  

test$pred.Transport = predict(gbm.fit, test.bal, type = "response")

table(test.bal$Transport,test.bal$pred.Transport>0.5)


train.bal$Gender = as.numeric(train.bal$Gender)
train.bal$Engineer = as.numeric(train.bal$Engineer)
train.bal$MBA = as.numeric(train.bal$MBA)
train.bal$Transport = as.numeric(train.bal$Transport)
train.bal$CustomerProfile = as.numeric(train.bal$CustomerProfile)
test.bal$Gender = as.numeric(test.bal$Gender)
test.bal$Engineer = as.numeric(test.bal$Engineer)
test.bal$MBA = as.numeric(test.bal$MBA)
test.bal$Transport = as.numeric(test.bal$Transport)
test.bal$CustomerProfile = as.numeric(test.bal$CustomerProfile)

train.bal$Gender[train.bal$Gender == 1] = 0
train.bal$Gender[train.bal$Gender == 2] = 1

test.bal$Gender[test.bal$Gender == 1] = 0
test.bal$Gender[test.bal$Gender == 2] = 1
 
train.bal$CustomerProfile[train.bal$CustomerProfile < 0.5] = 0
train.bal$CustomerProfile[train.bal$CustomerProfile > 0.5] = 1

test.bal$CustomerProfile[test.bal$CustomerProfile < 0.5] = 0
test.bal$CustomerProfile[test.bal$CustomerProfile > 0.5] = 1

train.bal$Transport[train.bal$Transport == 1] = 0
train.bal$Transport[train.bal$Transport == 2] = 1

test.bal$Transport[test.bal$Transport == 1] = 0
test.bal$Transport[test.bal$Transport == 2] = 1

str(train.bal)
str(test.bal)

summary(train.bal)
summary(test.bal)


features.train = as.matrix(train.bal)
label.train = as.matrix(train.bal$Transport)
features.test = as.matrix(test.bal)


XGtrain = xgb.DMatrix(data = features.train, label = label.train)




XGBmodel = xgboost(data = features.train , label = label.train,  eta = 0.1,
           max_depth = 3,
             nrounds = 10,
               nfold = 5,
                 objective = "binary:logistic",  
                   verbose = 0,   
                     early_stopping_rounds = 10)


XGBpredTest = predict(XGBmodel, features.test)
tabXGB = table(test.bal$Transport, XGBpredTest>0.5)
tabXGB
sum(diag(tabXGB))/sum(tabXGB)


