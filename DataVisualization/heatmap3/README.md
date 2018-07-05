# Heatmap 
- 작성사 : 김성민

## Heatmap3 사용법

## library
```
library(heatmap3) 
library(RColorBrewer) 
library(gplots)
```

## Read Data
```
od<-read.csv("D:/git/SungminCode/HGU/BioData/HowTo/heatmap/sample.csv",header = T,sep = ',') 
dim(od)
```

## Data Processing
```
data<-od
rownames(data)<-data$patient 
#data 의 row names을 data의 patient이름으로 data<-subset(data,select = -c(patient,index)) 
#heatmap3에는 index와 patient이름이 필요가 없다.
```

## Basic 
heatmap에 들어가는 data는 다 numeric data로 되어야 함.

그래서 cancer_code를 지우고, result를 gene interaction을 보는 heatmap 에는 필요가 없다. 

나중에 복잡한 heatmap 그릴때 필요!
