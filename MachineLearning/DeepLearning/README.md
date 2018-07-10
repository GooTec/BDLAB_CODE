# Deep Learning 
- 작성자 : 구태완
- e-mail : starymate@gmail.com

## Introduction
  이 문서는 바이오데이터랩 내에서 사용되는 딥러닝 코드 사용을 위한 문서입니다. 라이브러리 코드는 데이터 프로세싱을 위한 dataProcess.py와 딥러닝 모델 설정을 위한 deepLearning.py 2개의 파일이 있습니다. 이 2개의 라이브러리 파일을 이용해서 텐서플로우에 필요한 데이터 처리와 학습을 진행할 수 있습니다. 이 라이브러리 사용을 위한 예제코드는 ExampleCode.py로 이름을 붙여 두었습니다. 

## Installation
### Environment
- anaconda 3.6+ 
- Tensorflow 1.3.0+

### Installation guide
- anaconda 설치 : https://www.anaconda.com/download/   : windows, mac, Linux 각 OS에 맞게 python3.6+의 anaconda를 설치합니다. 이를 통해 기본적으로 필요한 모듈인 numpy, pandas , jupyter notebook을 일괄적으로 설치할 수 있습니다. 
- Tensorflow 설치 : https://www.tensorflow.org/install/  : anaconda가 정상적으로 설치되었다면 pip install tensorflow 명령을 통해 설치가 가능합니다. 
```sh
pip install tensorflow
```

## Input Data
- file name : GEO_input_ensemble_CV_500.csv
- 입력 설명
  * id : 특정 행을 식별할 수 있는 식별자
  * x1~xn : n개의 입력값 , 이 데이터를 통해 딥러닝 모델이 학습을 진행
  * result : 특정 행에 대한 실제 값 
  * index : n-fold를 진행하기 위한 group number

## 예제를 통한 Library 활용하기
### 모듈 읽어오기
```py
import lib.dataProcess as dp    #data processing을 위한 라이브러리
import lib.deepLearning as dl   #deep learning model 셋팅을 위한 라이브러리
import pandas as pd             #pandas 는 dataframe 모듈
import tensorflow as tf         #tensorflow 는 파이썬 기반의 딥러닝 모듈
```

```py
inputfile = input("Insert input file Name :") #입력 파일을 입력
```

## Test

