#BioData Lab  <the best model "SVM" defore deep learning >

#  by �輺��, 2018.6.22

#SVM�� ��� �ִ� libaray 

library(e1071)

#data�� �ҷ��´�.

data<-read.csv("D:/git/SungminCode/HGU/BioData/HowTo/heatmap/sample.csv",header = T,sep = ',')
data$result <-as.factor(data$result)
#Data processing for SVM
j = 1;
test <-data[data$index == j,]
train <-data[data$index != j, ]

test <- subset(test, select = -c(index,patient,cancer_code))  
train <- subset(train, select = -c(index,patient,cancer_code))

# SVM model

svm_model <- svm(result~.,data = train, kernel = "radial", cost = 1,coef.0 = 0.1 ,epsilon = 0.1)
svm_model

#test data set�� prediction model(svm_model)�� ����

pred <- predict(svm_model,test)
result_table<- table(pred,test$result)
result_table
auc <- sum(result_table[1,1],result_table[2,2])/sum(result_table)
auc
#confusionMatrix

library(caret) #confusion Matrix�� caret library �ȿ� �ִ�.

confusionMatrix(train$result,predict(svm_model))


#���߿� �ð��� �Ǹ� ������ parameter�� ã�� ���� tune.svm�� �̿�.
svm_tune<-tune.svm(result~.,data = train,kernel = 'radial',gamma = c(0.1,0.2),coef0 = c(0.1,0.5),cost = c(0.001,0.01))
svm_tune

#�� �ñ��Ѱ�.. ??SVM�� ����..
