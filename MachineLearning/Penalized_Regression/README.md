# Penalized Regression
- 작성자 : 한결희
- e-mail : 

## 파일 확인
C:\test 폴더를 만들고 그 안에 압축 파일을 모두 풀 것.
1) 공통
  - sample.csv(원본파일)
2) Penalized_Regression_simple.R(간단)
  - ref.csv(geneset 파일) 파일만 필요.
3) Penalized_Regression_Example.R(실제 사용)
  - ‘names’ 폴더 안의 파일들 필요.

## 기본 개념
1) Penalized Regression은 regression 기법의 한 종류로 Y = WX + B에서 weight에 제약 조건을 추가하는 방식으로 작동함. 상세한 내용은 아래 사이트 참조.
https://datascienceschool.net/view-notebook/83d5e4fff7d64cb2aecfd7e42e1ece5e/
본 코드에서는 Lasso, Ridge, Elastic Net 세 가지가 한꺼번에 돈다.
2) sample.csv파일을 열어보면 첫 번째 column이 환자 이름이고 그 뒤로 100개의 gene의 발현량이 기록돼 있다. 이후로 환자가 어떤 암인지 나타내는 cancer code, 환자가 암인지 나타내는 result(정상이면 0, 암이면 1), index(train & test dataset 나눌 때 사용.)가 있다. 본 코드의 목표는 result가 0인지 1인지 맞추는 것이다.
3) ref.csv를 열어보면 맨 위에 x가 있고 아래쪽에 여러 기호가 있다. 이 기호들은 gene name으로, 앞서 열어본 sample.csv 파일의 gene name 중 일부다. 여기서는 50개의 gene name이 있다. 맨 위의 x는 header다. 대부분의 regression은 전체 데이터셋을 사용하지 않고 특정 feature(본 데이터에서는 특정 gene)를 골라 학습하게 되는데, 어떤 feature를 쓸지 골라놓은 것이 이 ref 파일. 

## Penalized_Regression_simple.R
1) directory & file name setting
- input dir과 ref dir 설정. input dir은 분석에 쓸 sample.csv가 있는 디렉토리, ref dir은 gene set 레퍼런스 파일이 있는 곳. 압축파일을 앞서 언급한 디렉토리에 풀었다면 건드릴 필요가 없다.
- output dir 설정. 역시 건드릴 필요 없다. 나중에 accuracy들을 기록한 파일을 어디로 내보낼지 결정한다.
2) data file read
	- 연습에 쓸 데이터파일을 읽는다. sample.csv를 읽고 patient, cancer_code 등을 따로 떼어 저장한다.
	- genes의 경우 patient, result 등을 제외한 발현량 관련 table이다. 
3) library setting
	- library 관련 세팅이다. 패키지를 설치하지 않았다면 설치부터 해준다.
4) recorder initialization
	- 코드를 돌리면서 결과값을 저장할 벡터(vector) 변수다. r에는 레퍼런스 파일 이름을, method에는 regression 종류를, accuracy에는 정확도를 저장한다.
	- 나중에 이들을 묶어서 하나의 accuracy table을 만든다.
5) subsampling
	- ref를 읽어 들인다. read.csv를 통해 앞서 설명한 선별된 geneset을 읽으면 column이 하나인 table 형태로 데이터가 저장된다. 이를 vector 변수로 참조하고 character 변수로 바꿔준다.
	- subset 함수로 ref를 참조해 발현량 테이블을 subsampling한다. 
* dim(genes), dim(gene_sub)를 해보면 column 개수가 절반으로 줄어든 것을 확인해볼 수 있다.
	- cbind로 regression에 필요한 데이터 형태인 result – genes(subsampled)를 맞춰준다.
	- sample 함수로 전체 데이터의 20%를 test에, 나머지를 train에 사용한다.
	- mdlX, Y는 각각 train에 필요한 학습 데이터(발현량), result(암인지 정상인지. 0, 1)
	- newX, newY는 각각 test에 필요한 학습 데이터, result


6) regression
	- glmnet 패키지의 함수들을 사용해 regression model을 구성한다. 최종적으로 roc라는 일종의 모델 묶음이 나온다.
	- acc를 구한다. 이 때 warning message가 뜰 수 있으나 정상이다. best model이 하나 이상 있어서 생기는 경고 메시지다.
		* 여기서 구하는 것은 test accuracy. train을 구하고 싶으면 roc1에 newY, newX 대신 mdlX, mdlY를 넣을 것.
		* accuracy 말고 다른걸 구해보고 싶으면 acc1에서 ret=c(“accuracy”)부분에  여러 가지를 넣어보면 됨. specificity, sensitivity, tn 등…
	- accuracy와 reference, method를 각각 기록한다.
	- Lasso, Ridge, Elastic Net 모두 같은 과정을 거친다. Elastic Net의 경우 최적의 alpha값을 찾아야 하므로 좀 더 복잡하지만 이후 과정은 동일하다.
7) make acc table & save
	- 앞서 기록한 reference, method, accuracy 데이터를 data frame으로 만든다. 대상 디렉토리에 저장한다.

결과 파일

## Penalized_Regression_Example.R
실제 연구에 사용 중인 코드와 거의 비슷. 알아두면 유용하긴 하겠지만 굳이… 쓰고 나서야 왜 적었을까 후회하고 있음. 목표지향적인 삶을 살고 싶었는데. 기본적으로 cancer diagnosis 연구의 배경을 조금 설명. 앞 부분을 다 읽은 것을 전제로 새로 나온 부분만 설명.

1) Variance ~ Mean Selection
간단하게 말하면 어떤 기준에서 gene set을 설정할 것인지의 기준. 앞서 설명했던 ‘ref.csv’파일처럼 학습에 사용할 gene set을 설정하는 것은 매우 중요한 일임. variance가 높은 순서대로, 혹은 발현량 자체가 높은 순서대로 등 나름의 기준을 가지고 gene을 배열한 후 이를 앞에서부터 특정 개수(본 코드에서는 10개/30개/70개)만 뽑아서 모델에 이용하게 됨.
맨 위에 Get~과 Top~ 코드들은 본 코드에서 직접 쓰이지는 않지만 ref.csv와 같이 geneset을 뽑을 때 사용한 코드 모음. 상세한 설명은 아래와 같으며 바쁘면 스킵해도 무방.
(1) VAR (Variance) - 전체 환자 데이터에서 특정 gene의 variance를 구한 후 variance가 높은 순서대로 gene을 선별. 기본적으로 환자에 따라 발현량에 차이가 적은 gene보다 차이가 있는 gene이 데이터셋 구분에 중요하리라는 가설에 기반.
ex) 'VAR 400'이면 variance가 높은 gene 400개를 선별했다는 의미.
(2) CV (Coefficient of Variance) - 전체 환자 데이터에서 특정 gene의 coefficient of variation을 구한 후 CV가 높은 순서대로 gene을 선별. variance와 비슷한 가설이지만 각 환자마다 생기는 편차를 고려하기 위해 CV를 사용. 
ex) 'CV 100'이면 CV가 높은 gene 100개를 선별했다는 의미.
(3) Diff (Difference) - 라벨을 사용해 geneset을 선별. normal/cancer 환자군에서 각각 gene별로 발현량을 더하고, 이 둘을 빼고 절대값을 취해 각 환자군에서 발현량 차이가 가장 큰 geneset을 선별. 코드 읽어보면 좀 더 이해가 쉬울 것. 
ex) 'Diff 100'이면 sensitive와 resistant 환자군에서 발현량 차이가 가장 큰 gene 100개를 선별했다는 의미.
	* 이 밖에 원래 cancer에 있어 중요하다고 보고된 annotation geneset도 사용하는데 여기서는 설명 안 함. foundation308 & foundation2267이 그것들임.
2) ref_names
	- 앞서 설명한 simple 버전 코드와 달리 써야 할 reference 파일이 여러 개다.
- ref를 여러 개를 사용할 것이기 때문에(type별로, gene 개수별로) 공통적으로 앞에 붙어있는 이름인 "names_GEO_input_ensemble"만 일단 넣어놓고 나중에 개별적인 이름을 보여줄 것임.
3) types & features
- types는 각 gene을 뽑은 기준
ex) VAR면 전체 geneset을 variance가 높은 순으로 배열했다는 뜻
- features는 그래서 뽑은 gene의 개수
ex) type이 VAR, feature가 10이면 Variance 상위 10개 gene을 뽑았다는 뜻.
	- ref file을 만들 때 variance에 대해 10개, 30개, 70개 이런 식으로 뽑았기 때문에 types, features에 대해 2중 for문을 돌려 총 9개의 reference를 모두 사용할 것임.
4) recorder
	- 아까와 다른 부분은 t와 f인데, type과 feature를 뭘 썼는지 저장하는 부분임.
	- num은 for문 가장 안쪽을 돌 때마다 3씩 증가하는데, for문 한 번당 Ridge, Lasso, Elastic Net 세 개의 결과를 저장하기 때문에 그 위치를 지정해주는 것. 어디 쓰이는지 궁금하다면 Ctrl+F해서 ‘num’을 검색해볼 것.
5) type & feature
- for문이 본격적으로 시작. type과 feature는 각각 types와 features 벡터 변수에서ref_names한 개씩에 해당함. 궁금하면 for문 돌리면서 type과 feature를 출력하는 부분을 살펴볼 것.
- ref file은 실제 file 이름. 압축파일 내의 names 디렉토리(설명대로 풀었다면 C:\\test\\names)에서 reference 파일 이름을 보면 “names_GEO_input_ensemble_CV_10.csv” 이런 식으로 앞의 공통 부분(앞서 ref_names로 정의)과 type(여기서는 CV), feature 개수(여기서는 10), 마지막 공통 부분 “.csv”로 나뉜다. 이를 고대로 넣어줌. 
* 궁금하면 for문 돌리면서 출력하는 “ref is ~” 문장을 참조.
- ref는 위에서 정의한 ref_file에 “.csv”를 더해서 실제 파일의 이름을 지칭하고 이를 read.csv로 읽고… 기타 등등 한 파일. 실제 gene 이름이 들어간 vector 변수임.
- gene_sub는 genes를 ref 파일에 있는 gene으로 선택해 자른 것.
6) cross validation
	- 앞서 원본 데이터를 train & test로 분리할 때 sample함수로 랜덤하게 뽑았는데, 여기서는 모델의 안정성을 평가하기 위해 index를 사용함. 전체 데이터셋을 다섯 덩이로 나누고 한 덩어리는 테스트에, 나머지 네 덩어리는 학습에 사용하는 방식. 어떤 것을 test에 쓸 것인지에 따라 다섯 번 학습&테스트를 거치게 됨.
	- 실제 regression은 거의 비슷. ref file 이름을 통째로 기록하는 것이 아니라 type, feature로 나눠서 기록하기 때문에 분류가 좀 더 간편(ex: CV만 찾기, 30개짜리들 비교하기…)

결과 예시

사족1) 모르는 부분이 있다면 작성자도 모를 확률이 다분하므로 교수님과 구글님께 여쭤보고, 틀린 부분이 있다면 꼭 알려주세요 제발. 연구 다시 해야 되니까.
사족2) 원본 코드를 보내주신 이성영 박사님께 심심한 감사를…
사족3) 모두 파이팅
