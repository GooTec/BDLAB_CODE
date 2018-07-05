import tensorflow as tf
import numpy as np

'''구분해서 사용할 수 있게 해야 하는 내용 activation function'''

def get_activation_fn(input_tensor, activation_fn='relu'):
    try :
        if activation_fn == 'relu' :
            return tf.nn.relu(input_tensor, 'relu')
        elif activation_fn == 'sigmoid':
            return tf.sigmoid(input_tensor, 'sigmoid')
        elif activation_fn =='tanh':
            return tf.tanh(input_tensor, 'tanh')
        else:
            raise TypeError
    except TypeError as TE:
        print("Wrong Activation Function ")
        raise
    except :
        raise



def dense(x, size, scope, activation_fn):
    '''
    :param x: 입력 텐서
    :param size: 출력으로 나와야하는 텐서의 크기
    :param scope: 텐서의 이름 영역
    :return: fully connected neural network 레이어

    Default value of fully_connected function
    (
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
    )
    '''
    return tf.contrib.layers.fully_connected(x, size, activation_fn=activation_fn,scope=scope )

def dense_batch(x, phase, size, scope, activation_fn='relu'):
    '''
    :param x: 입력 텐서
    :param phase: 현재 텐서플로우가 학습 중인지 테스트 중인지를 구분한다 phase가 True이면 학습 False이면 테스트
    :param size: 출력으로 나와야하는 텐서의 크기
    :param scope: 텐서의 이름 영역
    :return: fully connected neural network - batch normalization - relu가 적용 된 레이어
    '''
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size,activation_fn=None,scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase,scope='bn')
        return get_activation_fn(h2, activation_fn)


def dense_dropout(x,phase, size, scope, rate=0.5, activation_fn='relu'):
    '''
    :param x: 입력 텐서
    :param phase: 현재 텐서플로우가 학습 중인지 테스트 중인지를 구분한다 phase가 True이면 학습 False이면 테스트
    :param size: 출력으로 나와야하는 텐서의 크기
    :param scope: 텐서의 이름 영역
    :param rate: 드랍아웃이 적용 비
    :return: fully connected neural network - drop out - relu가 적용 된 레이어
    '''
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope='dense')
        h2 = tf.contrib.layers.dropout(h1, keep_prob=rate , is_training = phase ,scope='do')
        return get_activation_fn(h2, activation_fn)



def dense_batch_dropout(x,phase, size, scope, rate=0.5, activation_fn='relu'):
    '''
    :param x: 입력 텐서
    :param phase: 현재 텐서플로우가 학습 중인지 테스트 중인지를 구분한다 phase가 True이면 학습 False이면 테스트
    :param size: 출력으로 나와야하는 텐서의 크기
    :param scope: 텐서의 이름 영역
    :param rate: 드랍아웃이 적용 비
    :return: fully connected neural network - drop out - relu가 적용 된 레이어
    '''
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase,scope='bn')
        h3 = tf.contrib.layers.dropout(h2, keep_prob=rate , is_training = phase ,scope='do')
        return get_activation_fn(h3, activation_fn)


def set_model_dropout(x, y, nodes , learning_rate, activation_fn ='relu'):
    '''
    :param x: 입력 데이터
    :param y: 기댓값
    :param nodes: 네트워크 레이어의 노드 갯수를 담고 있는 리스트
    :param learning_rate: 학습 시 사용 될 learning rate
    :return: 모델에 사용되는 변수들
    '''
    tf.reset_default_graph()


    np.random.seed(777)
    keep_prob = tf.placeholder(tf.float32) #0.7 -> 70% 켜진 채로 30% 꺼진 채로
    layers = []

    X = tf.placeholder(tf.float32, [None, len(x[0])])
    Y = tf.placeholder(tf.float32, [None, len(y[0])])

    phase = tf.placeholder(tf.bool, name='phase')

    for i in range(len(nodes)):
        if i == 0:
            layers.append(dense_dropout(X, phase, nodes[i], 'layer'+str(i+1), keep_prob, activation_fn))
        else:
            layers.append(dense_dropout(layers[i-1], phase, nodes[i], 'layer'+str(i+1), keep_prob, activation_fn))

    logits = dense(layers[-1], len(y[0]), 'logits', None)
    hypothesis = tf.nn.softmax(logits)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal( predicted, tf.argmax( Y, 1))
    accuracy = tf.reduce_mean(tf.cast( correct_prediction, dtype=tf.float32))

    return X , Y , layers, logits  ,phase, hypothesis, cost , train, predicted , correct_prediction , accuracy , keep_prob


def set_model_BN(x, y, nodes, learning_rate, activation_fn='relu') :
    tf.reset_default_graph()

    np.random.seed(777)
    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, [None, len(x[0])])
    Y = tf.placeholder(tf.float32, [None, len(y[0])])
    phase = tf.placeholder(tf.bool, name='phase')
    layers = []

    for i in range(len(nodes)):
        if i == 0:
            layers.append(dense_batch(X, phase, nodes[i], 'layer'+str(i+1), activation_fn))
        else :
            layers.append(dense_batch(layers[i-1], phase, nodes[i], 'layer'+str(i+1), activation_fn))
    # h1 = dense_batch_relu(X, phase, nodes[0], 'layer1')
    # h2 = dense_batch_relu(h1, phase, nodes[1],'layer2')
    # h3 = dense_batch_relu(h2, phase, nodes[2],'layer3')
    # h4 = dense_batch_relu(h2, phase, nodes[3],'layer4')
    # layers.append(h1)
    # layers.append(h2)
    # layers.append(h3)
    # layers.append(h4)
    logits = dense(layers[-1], len(y[0]), 'logits', None)
    hypothesis = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False

    predicted = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal( predicted, tf.argmax( Y, 1))
    accuracy = tf.reduce_mean(tf.cast( correct_prediction, dtype=tf.float32))

    return X , Y , layers, logits, phase  , hypothesis, cost , train, predicted , correct_prediction , accuracy , keep_prob


def set_model_batch_dropout(x, y, nodes , learning_rate, activation_fn='relu'):
    '''
    :param x: 입력 데이터
    :param y: 기댓값
    :param nodes: 네트워크 레이어의 노드 갯수를 담고 있는 리스트
    :param learning_rate: 학습 시 사용 될 learning rate
    :return: 모델에 사용되는 변수들
    '''
    tf.reset_default_graph()


    np.random.seed(777)
    keep_prob = tf.placeholder(tf.float32) #0.7 -> 70% 켜진 채로 30% 꺼진 채로
    layers = []

    X = tf.placeholder(tf.float32, [None, len(x[0])])
    Y = tf.placeholder(tf.float32, [None, len(y[0])])

    phase = tf.placeholder(tf.bool, name='phase')

    for i in range(len(nodes)):
        if i == 0:
            layers.append(dense_batch_dropout(X, phase, nodes[i], 'layer'+str(i+1), keep_prob, activation_fn))
        else:
            layers.append(dense_batch_dropout(layers[i-1], phase, nodes[i], 'layer'+str(i+1), keep_prob, activation_fn))

    logits = dense(layers[-1], len(y[0]), 'logits', None)
    hypothesis = tf.nn.softmax(logits)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal( predicted, tf.argmax( Y, 1))
    accuracy = tf.reduce_mean(tf.cast( correct_prediction, dtype=tf.float32))

    return X , Y , layers, logits  ,phase, hypothesis, cost , train, predicted , correct_prediction , accuracy , keep_prob

def set_model_basic(x, y, nodes , learning_rate, activation_fn):
    '''
    :param activation_fn: 모델의 히든 레이어에 사용 될 activation function을 지정
    :return:
    '''
    tf.reset_default_graph()


    np.random.seed(777)
    keep_prob = tf.placeholder(tf.float32) #0.7 -> 70% 켜진 채로 30% 꺼진 채로
    layers = []
    phase = tf.placeholder(tf.bool, name='phase')

    X = tf.placeholder(tf.float32, [None, len(x[0])])
    Y = tf.placeholder(tf.float32, [None, len(y[0])])

    phase = tf.placeholder(tf.bool, name='phase')
    try :
        activation_fn = activation_fn.lower()
        if activation_fn =='relu':
            activation_fn = tf.nn.relu
        elif activation_fn == 'sigmoid':
            activation_fn = tf.sigmoid
        elif activation_fn == 'tanh':
            activation_fn = tf.tanh
        else :
            raise ValueError
    except ValueError :
        print("WRONG ACTIVATION FUNCTION NAME.\n"
              "YOU CAN USE ACTIVATION FUNCTIONS : \n"
              "1) relu\n"
              "2) sigmoid\n"
              "3) tanh")

    for i in range(len(nodes)):
        if i == 0:
            layers.append(dense(X, nodes[i], 'layer'+str(i+1), activation_fn))
        else:
            layers.append(dense(layers[i-1],  nodes[i], 'layer'+str(i+1), activation_fn))

    logits = dense(layers[-1], len(y[0]), 'logits', None)
    hypothesis = tf.nn.softmax(logits)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal( predicted, tf.argmax( Y, 1))
    accuracy = tf.reduce_mean(tf.cast( correct_prediction, dtype=tf.float32))

    return X , Y , layers, logits  , phase,  hypothesis, cost , train, predicted , correct_prediction , accuracy , keep_prob


def set_model_by_paramter(x, y, nodes, learning_rate, drop_out=True, batch_norm=False, activation_fn='relu'):
    '''
    tf.nn.relu
    tf.nn.relu6
    tf.nn.crelu
    tf.nn.elu
    tf.nn.softplus
    tf.nn.softsign
    tf.nn.dropout
    tf.nn.bias_add
    tf.sigmoid
    tf.tanh
    '''
    if activation_fn == 'relu' :
        if drop_out and batch_norm:
            return set_model_batch_dropout(x,y,nodes,learning_rate )
        elif drop_out :
            return set_model_dropout(x, y, nodes, learning_rate)
        elif batch_norm :
            return set_model_BN(x,y,nodes,learning_rate )
        else :
            return set_model_basic(x,y,nodes,learning_rate,'relu')
    elif activation_fn == 'sigmoid' :
        if drop_out and batch_norm:
            return set_model_batch_dropout(x,y,nodes,learning_rate, 'sigmoid')
        elif drop_out :
            return set_model_dropout(x, y, nodes, learning_rate, 'sigmoid')
        elif batch_norm :
            return set_model_BN(x,y,nodes,learning_rate, 'sigmoid')
        else :
            return set_model_basic(x,y,nodes,learning_rate,'sigmoid')
    elif activation_fn == 'tanh' :
        if drop_out and batch_norm:
            return set_model_batch_dropout(x,y,nodes,learning_rate, 'tanh')
        elif drop_out :
            return set_model_dropout(x, y, nodes, learning_rate, 'tanh')
        elif batch_norm :
            return set_model_BN(x,y,nodes,learning_rate, 'tanh')
        else :
            return set_model_basic(x,y,nodes,learning_rate,'tanh')
    else :
        return set_model_dropout(x, y, nodes, learning_rate)
