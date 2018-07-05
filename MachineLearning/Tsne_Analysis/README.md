# TSNE Analysis
- 작성자 : 함현정
- E-mail : 

## 예제 데이터
- R에서 기본적으로 가지고 있는 데이터 셋인 iris 이용 

## 샘플 코드
```R
#install.packages(“tsne”) #패키지 설치 
	library(tsne)
colors = rainbow(length(unique(iris$Species)))
#데이터를 범주별로 구분해서 서로 다른 색으로 그리기 위해서 설정해준다. 
names(colors) = unique(iris$Species)
#각 색이 어떤 범주를 가리키는지를 알기 위해서 이름을 설정해준다. 
ecb = function(x,y){ plot(x,t='n'); points(x, pch = 20, col=colors[iris$Species]) }
#tsne가 돌아가면서 반복할 callback 함수를 지정해 준다. 여기서는 점 그래프를 그린다. 
tsne_iris = tsne(iris[,1:4], epoch_callback = ecb, perplexity=50, initial_dims = 3, k = 2)
```

## 사용법
```R
tsne(X, initial_config = NULL, k = 2, initial_dims = 30, perplexity = 30,
max_iter = 1000, min_cost = 0, epoch_callback = NULL, whiten = TRUE,
epoch=100)
```

## Argument
```R
X: The R matrix or "dist" object
initial_config: an argument providing a matrix specifying the initial embedding for X. 
#각 epoch이 지날 때마다 얼마나 조정할지를 결정한다고 한다. (사용해본 적 없음)
k: the dimension of the resulting embedding.
#embedding 결과 몇 개의 차원으로 나타낼 지를 입력한다. 2차원 이상으로 표현할 때는 해당 차원을 표현할 수 있는 그래프를 그려야 한다. (보통 2차원)
initial_dims: The number of dimensions to use in reduction method.
#첫 embedding의 차원을 입력한다. Ex)유전자/패스웨이 개수 
Perplexity: Perplexity parameter. (optimal number of neighbors)
#적정 이웃의 개수를 입력한다. 
max_iter: Maximum number of iterations to perform.
#최대 몇 번의 iteration을 거칠 것인지를 입력한다. 
min_cost: The minimum cost value (error) to halt iteration.
#특정 error이하 값이 나오면 iteration을 멈추게 한다. 
epoch_callback: A callback function used after each epoch (an epoch here means a set number of iterations)
#epoch에서 설정한 값마다 수행할 작업을 function으로 입력한다. 
whiten: A boolean value indicating whether the matrix data should be whitened.
#각 epoch에서 설정한 값마다 matrix data를 지울지 수정할지를 정한다. 
epoch: The number of iterations in between update messages.
#몇 epoch마다 update message를 출력할지를 정한다. 
```





