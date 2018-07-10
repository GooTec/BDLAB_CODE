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
### 입력 파일 읽어오기
```py
inputfile = input("Insert input file Name :") #입력 파일을 입력
raw_data = pd.read_csv(inputfile)             #read_csv() 함수를 이용해 csv file을 읽는다.
```

### n-fold위한 데이터 분리
```py
fivefold = dp.n_fold(raw_data, 'index', 5)    #parameter 설명(입력 데이터, 'n-fold를 위한 그룹정보, n)
```

### Train, Test, Validation Data set 분리 작업
```py
xdata_five, ydata_five = dp.divide_xy_train(fivefold, 'result', True, 1, -3)    #Data를 xdata와 ydata 로 분리한다.
train_x, test_x = dp.train_test(xdata_five, 0)                #train, test data 분리 
train_y, test_y = dp.train_test(ydata_five, 0)                #train, test data 분리
train_y = dp.one_hot_encoder(train_y)                         #train y를 one hot encoding
test_y = dp.one_hot_encoder(test_y)                           #test y를 one hot encoding
val_x, test_x = dp.test_validation(test_x)                    #validation, test data 분리
val_y, test_y = dp.test_validation(test_y)                    #validation, test data 분리
```

### 모델 셋팅
```py
nodes = [200,100, 50]     #모델의 레이어별 노드 갯수 
learning_rate = 0.001     #모델의 학습 속도 조절
batch_size = 100          #모델이 학습하는 batch의 크기 설정  

X, Y, layers, logits, phase, hypothesis, cost, train, predicted, correct_prediction, accuracy, keep_prob = dl.set_model_dropout(train_x, train_y, nodes , learning_rate)  #dl library의 함수를 이용해 모델 셋팅
```

### 모델 실행
```py
saver = tf.train.Saver()      #모델을 저장하고 불러오기 위한 saver 객체
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())     #변수 초기화
    stop_switch = True        #학습의 종료를 위한 변수
    step = 0                  #학습 횟수

    while stop_switch: 
        total_num = int(len(train_x) / batch_size)
        '''
        이 for loop를 통해 batch 사이즈만큼씩 학습을 진행한다. 
        for loop를 1번 돌고 나면 1 epoch 
        '''
        for k in range(total_num):
            batch_x = train_x[k * batch_size:(k + 1) * batch_size]
            batch_y = train_y[k * batch_size:(k + 1) * batch_size]
            sess.run(train, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5 , phase:True})
  
        
        '''
        train data와 validation data의 현재 모델 결과를 얻는 코드
        이 결과를 통해 학습을 계속 진행할 것인지를 결정
        '''
        train_h, train_c, train_p, train_a = sess.run( [hypothesis, cost, predicted, accuracy], feed_dict={X: train_x, Y: train_y, keep_prob: 1 , phase:False})
        val_h, val_c, val_p, val_a = sess.run([hypothesis, cost, predicted, accuracy], feed_dict={X: val_x, Y: val_y, keep_prob: 1 , phase:False})
        if step % 20 == 0 :
            print("train acc : ", train_a, "validation acc : ", val_a, "train_cost", train_c)
        step += 1

        
        '''
        학습의 종료를 위한 조건문
        '''
        if best_cost > val_c :
            best_train_acc = train_a
            best_val_acc = val_a
            best_cost = val_c
            count = 0
            save_path = saver.save(sess, save_path_dir + 'model'+str(model_num)+'.ckpt')

        elif count > 10 :
            stop_switch = False
            print("Learning Finished!! \n")
        else:
            count += 1
    
    '''학습이 종료된 후에 실행되는 코드'''
    print("Training Accuracy : ", best_train_acc,  "Validation Accuracy : ", best_val_acc)
    
    saver.restore(sess, save_path)      #현재까지 최고의 모델을 다시 불러온다.


    test_h, test_p, test_a = sess.run([hypothesis, predicted, accuracy],
                                      feed_dict={X: test_x, Y: test_y, keep_prob:1.0 , phase:False})
    print("\nTest Accuracy: ", test_a)    #테스트 데이터셋의 정확도를 출력한다.
    best_test_acc = test_a


    model_num += 1                    
    
```



## Test

