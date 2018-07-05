{
  #################################################### VarianceTest ####################################################
  GetVar<-function(genes){
    VAR<-apply(genes,2,sd)
    return(VAR)
  }
  
  TopVar <- function(genes, feature){
    VAR <- GetVar(genes)
    gene_ch <- rbind(genes, VAR)
    gene_ch_VAR <- gene_ch[,rev(order(gene_ch[nrow(gene_ch),]))]
    gene_sub <-gene_ch_VAR[-nrow(gene_ch_VAR), 1:feature]
    return(gene_sub)
  }
  
  #################################################### DiffTest ####################################################
  GetDiff <- function(genes, result){
    negative <- apply(genes[result==0,],2,mean)
    positive <- apply(genes[result==1,],2,mean)
    Diff <- abs(positive - negative)
    return(Diff)
  }
  
  TopDiff <- function(genes, result, feature){
    Diff <- GetDiff(genes, result)
    gene_ch <- rbind(genes, Diff)
    gene_ch_Diff <- gene_ch[,rev(order(gene_ch[nrow(gene_ch),]))]
    gene_sub <-gene_ch_Diff[-nrow(gene_ch_Diff), 1:feature]
    return(gene_sub)
  }
  
  #################################################### Coefficient of Variance ##########################################
  GetCV <- function(genes){
    CV<-apply(genes, 2, sd)/apply(genes, 2, mean)
    return(CV)
  }
  
  TopCV <- function(genes, feature){
    CV <- GetCV(genes)
    gene_ch <- rbind(genes, CV)
    gene_ch_CV <- gene_ch[,rev(order(gene_ch[nrow(gene_ch),]))]
    gene_sub <-gene_ch_CV[-nrow(gene_ch_CV), 1:feature]
    return(gene_sub)
  }
  
  #################################################### Mean Selection ####################################################
  GetMean<-function(genes){
    MEAN<-apply(genes,2,mean)
    return(MEAN)
  }
  
  TopMean <- function(genes, feature){
    MEAN <- GetMean(genes)
    gene_ch <- rbind(genes, MEAN)
    gene_ch_MEAN <- gene_ch[,rev(order(gene_ch[nrow(gene_ch),]))]
    gene_sub <-gene_ch_MEAN[-nrow(gene_ch_MEAN), 1:feature]
    return(gene_sub)
  }
}
#################################################### Main ####################################################

######################### Guide 
# input: a name of raw data file.
# ref: a name of selected gene set file.  
# output: a name of regression csv file.
# Add '/' to the end of the directory name.
input_dir <- "C:\\test\\"
input_file <- "sample.csv"
ref_dir <- "C:\\test\\names\\"
ref_names <- "names_GEO_input_ensemble"
output_dir <- "C:\\test\\"
output_names <- "GEO_penalizaed_regression.csv"

input <- paste0(input_dir, input_file)
df <- read.csv(input, header=TRUE)
patient <- df$patient
genes <- df[,setdiff(colnames(df), c("patient","result","cancer_code","index"))]
cancer_code <- df$cancer_code
result <- df$result
index <- df$index

print(paste0("subsampling ready. for confirm, first: ", names(genes)[1]))
print(paste0("subsampling ready. for confirm, last: ", names(genes)[ncol(genes)]))

types<- c("VAR", "Mean", "CV"
          #, "foundation_308", "foundation_2267"
          )
features <- c(10,30,70)

#install.packages("doParallel")
#install.packages("foreach")
#install.packages("pROC")
#install.packages("glmnet")
library("doParallel")
library("foreach")
library("pROC")
library("glmnet")

pkgs <- list("glmnet", "doParallel", "foreach", "pROC")
lapply(pkgs, require, character.only = T)
registerDoParallel(cores = 4)

#recorder initialization
{
  t<-NULL
  f<-NULL
  method <- NULL
  accuracy <- NULL
  acc_table <- NULL
  num <- 1
}

for(type in types){
  
  print(paste0("type: ", type))
  for(feature in features){
    #ref file read & subsampling
    if(type=="foundation_308" || type=="foundation_2267"){
      # ref_file <- paste(ref_names, type, sep="_")
      # if(feature!=500){
      #   break()
      # }
    }else{
      ref_file <- paste(ref_names, type, as.character(feature), sep="_")
    }
    print(paste0("feature: ", as.character(feature)))
    ref <- as.character(read.csv(paste0(ref_dir,ref_file, ".csv"), header = TRUE)$x)
    gene_sub <- subset(genes, select=ref)
    print(paste0("subsampled. ref is ", ref_file))
    print("Cross Validation starts.")
    df1 <- cbind(result, gene_sub)
    
    for(i in 1:5){
      print(paste0("i is ", as.character(i)))
      train <- df1[index!=i, ]
      test <- df1[index==i, ]
      #five-fold data set cross validation
      
      mdlY <- as.factor(as.matrix(train["result"]))
      mdlX <- as.matrix(train[setdiff(colnames(df1), "result")])
      newY <- as.factor(as.matrix(test["result"]))
      newX <- as.matrix(test[setdiff(colnames(df1), "result")])
      
      ######## LASSO WITH ALPHA = 1
      cv1 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 1)
      md1 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv1$lambda.1se, alpha = 1)
      roc1 <- roc(newY, as.numeric(predict(md1, newX, type = "response")))
      acc1 <- ci.coords(roc1, x="best", input = "threshold", ret=c("accuracy"), best.policy = "omit")
      accuracy <- c(accuracy,acc1[2])
      
      method <- c(method, "LASSO")
      
      ######## RIDGE WITH ALPHA = 0
      cv2 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 0)
      md2 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv2$lambda.1se, alpha = 0)
      roc2 <- roc(newY, as.numeric(predict(md2, newX, type = "response")))
      acc2 <- ci.coords(roc2, x="best", input = "threshold", ret=c("accuracy"), best.policy = "omit")
      accuracy <- c(accuracy,acc2[2])
      
      method <- c(method, "RIDGE")
      
      ######## ELASTIC NET WITH 0 < ALPHA < 1
      a <- seq(0.1, 0.9, 0.05)
      #hyperparameter
      search <- foreach(i = a, .combine = rbind) %dopar% {
        cv <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = i)
        data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
      }
      cv3 <- search[search$cvm == min(search$cvm), ]
      md3 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv3$lambda.1se, alpha = cv3$alpha)
      roc3 <- roc(newY, as.numeric(predict(md3, newX, type = "response")))
      acc3 <- ci.coords(roc3, x="best", input = "threshold", ret=c("accuracy"), best.policy = "omit")
      accuracy <- c(accuracy,acc3[2])
      
      method <- c(method, "EN")
      
      print(paste("acc(LASSO, RIDGE, EN): ", accuracy[num], accuracy[num+1], accuracy[num+2],sep="  "))
      t <- c(t, rep(type, 3))
      f <- c(f, rep(feature, 3))
      
      num <- num+3
    }
  }
}

acc_table <- data.frame(type = t, feature = f, method, accuracy)
output_file <- paste0(output_dir, output_names)
write.csv(acc_table, output_file, row.names = FALSE)
