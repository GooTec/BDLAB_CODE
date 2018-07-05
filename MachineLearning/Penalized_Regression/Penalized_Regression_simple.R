#################################################### Main ####################################################

######################### Guide 
# input: a name of raw data file.
# ref: a name of selected gene set file.  
# output: a name of regression csv file.
# Add '/' to the end of the directory name.

# 1) directory & file name setting
input_dir <- "C:\\test\\"
input_file <- "sample.csv"
ref_dir <- "C:\\test\\"
ref_names <- "ref"
output_dir <- "C:\\test\\"
output_names <- "GEO_penalizaed_regression_simple.csv"
input <- paste0(input_dir, input_file)
ref_file <- paste0(ref_dir, ref_names)

# 2) data file read
df <- read.csv(input, header=TRUE)
patient <- df$patient
genes <- df[,setdiff(colnames(df), c("patient","result","cancer_code","index"))]
cancer_code <- df$cancer_code
result <- df$result
index <- df$index

# 3) library setting
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

# 4) recorder initialization
{
  r <- NULL
  method <- NULL
  accuracy <- NULL
}

{
  # 5) subsampling
  ref <- as.character(read.csv(paste0(ref_file, ".csv"), header = TRUE)$x)
  gene_sub <- subset(genes, select=ref)
  #dim(genes)
  #dim(gene_sub)
  print(paste0("subsampled. ref is ", ref_names))
  df1 <- cbind(result, gene_sub)
  n <- nrow(df1)
  sample <- sample(seq(n), size = n * 0.2, replace = FALSE)
  train <- df1[-sample,]
  test <- df1[sample,]
  #sampling 20% of extracted samples for test, 80% for train.
  
  mdlY <- as.factor(as.matrix(train["result"]))
  mdlX <- as.matrix(train[setdiff(colnames(df1), "result")])
  newY <- as.factor(as.matrix(test["result"]))
  newX <- as.matrix(test[setdiff(colnames(df1), "result")])
  
  # 6) regression
  ######## LASSO WITH ALPHA = 1
  cv1 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 1)
  md1 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv1$lambda.1se, alpha = 1)
  roc1 <- roc(newY, as.numeric(predict(md1, newX, type = "response")))
  acc1 <- ci.coords(roc1, x="best", input = "threshold", ret=c("accuracy"), best.policy = "omit")
  accuracy <- c(accuracy,acc1[2])
  r <- c(r, ref_names)
  method <- c(method, "LASSO")
  print(paste0("ref: ", ref_names, ", LASSO accuracy is ", acc1[2]))
  
  ######## RIDGE WITH ALPHA = 0
  cv2 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 0)
  md2 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv2$lambda.1se, alpha = 0)
  roc2 <- roc(newY, as.numeric(predict(md2, newX, type = "response")))
  acc2 <- ci.coords(roc2, x="best", input = "threshold", ret=c("accuracy"), best.policy = "omit")
  accuracy <- c(accuracy,acc2[2])
  r <- c(ref_names, ref)
  method <- c(method, "RIDGE")
  print(paste0("ref: ", ref_names, ", RIDGE accuracy is ", acc2[2]))
  
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
  r <- c(r, ref_names)
  method <- c(method, "EN")
  print(paste0("ref: ", ref_names, ", EN accuracy is ", acc3[2]))
}

# 7) make acc table & save 
acc_table <- data.frame(ref = r, method, accuracy)
output_file <- paste0(output_dir, output_names)
write.csv(acc_table, output_file, row.names = FALSE)

