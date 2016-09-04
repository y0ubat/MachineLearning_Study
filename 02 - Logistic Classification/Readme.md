#ML - Logistic Regression

##Binnary Classification

이전의 Regression은 어떤 숫자를 예측하는 것이라면, Binnary Classification은 둘 중 하나를 고르는 겁니다.

*Example*

- Spam Detection: Spam(1) or Ham(0)
- Facebook feed: show(1) or hide(0)
<br>
<br>

|x (hours)|y (pass or fail)|
|:------: |:-------------:|
|   2     |     fail	   |
|   4     |     fail      |
|   5     |     pass      |
|   6     |     pass      |
|  50     |     pass      |

이러한 Train Data로 Linear Regression으로 구현했을 때, 몇 시간 이상 공부한 사람부터는 pass라고 예측하겠죠?<br>
여기서 문제가 생깁니다. Pass/Fail로 2개로 나눠집니다.
상대적으로 많은 시간(50시간)을 공부하고 통과한 사람이 있기 때문에, Train Data에는 5시간이 통과임에 불구하고 결과를 출력해보면, fail라고 뜨는 문제가 생깁니다.
<br>

	import tensorflow as tf
 
	x_data = [1.,5.,7.,10.,50.]
	y_data = [0.,0.,1.,1.,1.]

	W = tf.Variable(tf.random_uniform([1],-5.0,5.0))

	X = tf.placeholder(tf.float32)
	Y = tf.placeholder(tf.float32)

	h = W*X
	cost = tf.reduce_mean(tf.square(h-Y))

	a = tf.Variable(0.001)
	optimizer = tf.train.GradientDescentOptimizer(a)
	train = optimizer.minimize(cost)

	init = tf.initialize_all_variables()

	sess = tf.Session()
	sess.run(init)

	for step in xrange(2000):
    	sess.run(train,feed_dict={X:x_data,Y:y_data})
    	if step % 100 ==0:
        	print step,sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run(W)


	print sess.run(h,feed_dict={X:7})>0.5
	
이렇게 한번 Linear Regression으로 짜서 한번 돌려보면, Train Data는 7시간 공부하면 Pass로 되어 있는데 결과를 보면 Fail이 나오는 것을 볼 수 있습니다.

***Result***
	
	~생략~
	1800 0.264374 [ 0.02504673]
	1900 0.264374 [ 0.02504673]
	[False]
	

Linear Regression에 사용된 `H(x)=Wx+b`도 문제가 있습니다.<br>
0~1 사이로 값이 나와야 되는데, x의 값이 커지면 1 이상의 값으로 나와버리기 때문입니다.
<br>
그러므로 H(x)를 0~1사이의 값이 나오는 함수로 수정해줘야 됩니다.<br><br>
![](http://i.imgur.com/nw6PWN8.png) <br>이 함수를 그래프로 그려보면, 0~1사이 값을 가지는 그래프가 나옵니다.
<br><br>
![](http://i.imgur.com/pE5jMUTm.png)
<br><br>
이러한 함수를 sigmoid function, Logistic function이라고 합니다.<br><br>
![](http://i.imgur.com/qT5U5dTm.png)<br>
`H(x)=Wx+b`를 저 함수에 적용시키면, 이렇게 됩니다.
<br>
이 H(x)로 Cost Function의 그려보면 이렇게 울퉁불퉁해 집니다.
<br><br>
![](http://i.imgur.com/kfFdzaPm.png)
<br>
그래서 Gradient Descent 알고리즘을 사용하지 못하게 됩니다.<br>
그러므로 Cost Function을 조금 변경해줘야 됩니다.<br><br>
![](http://i.imgur.com/ztC2fQom.png) <br>
g(z)에서 분모에 e^-z 부분이 울퉁불퉁해지가 해주는 역할을 하는데요.
이것을 피기 위해서는 e^-z와 상극이 되는 log를 사용합니다<br><br>
![](http://i.imgur.com/fRrnKFZm.png) <br>
< `-log(H(x))` : y = 1 ><br><br>
![](http://i.imgur.com/x8sgCw5m.png) <br>
< `-log(1-H(x))` : y = 0 ><br><br>
다시 이렇게 매끄럽게 그래프가 그려졌기 때문에, Gradient Descent 알고리즘을 사용할 수 있습니다.<br><br>
![](http://i.imgur.com/zUjgnbvm.png)
<br>
![](http://i.imgur.com/7OjwG4Am.png)
<br>C(H(x),y)를 좀 정리하면, 이런 식으로 됩니다.<br>
Cost Functino과 H(x)가 조금 달라진 거 외에는 Linear Regression과 똑같습니다.<br>


##Logistic Classification 구현

Tensorflow 라이브러리를 이용해서 구현했습니다.<br>
전에 Linear Regression과 별 차이는 없고요, 위에 설명한 수식으로 수정만 해주면 됩니다.

	import tensorflow as tf
	import numpy as np

	xy = np.loadtxt('train.txt',unpack=True,dtype='float32')
	x_data = xy[0:-1]
	y_data = xy[-1]

	X = tf.placeholder(tf.float32)
	Y = tf.placeholder(tf.float32)

	W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))

	h = tf.matmul(W,X)
	h2 = tf.div(1.,1.+tf.exp(-h))

	cost = -tf.reduce_mean(Y*tf.log(h2) + (1-Y)*tf.log(1-h2))

	a = tf.Variable(0.1)
	optimizer = tf.train.GradientDescentOptimizer(a)
	train = optimizer.minimize(cost)

	init = tf.initialize_all_variables()

	sess = tf.Session()
	sess.run(init)

	for step in xrange(200000):
    	sess.run(train,feed_dict={X:x_data,Y:y_data})
    	if step % 1000 == 0:
       		print step, sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(W)

	print sess.run(h2,feed_dict={X:[[1],[10],[1]]})
	print sess.run(h2,feed_dict={X:[[1],[4],[4]]})
	
<br>
***Train.txt***

	#b x1 x2	y
	1	2	1	0
	1	3	2	0
	1	3	4	0
	1	5	5	1
	1	7	5	1
	1   2   5   1
<br>

##컴파일 결과

***Result***

	150000 0.00582836 [[-40.30181122   0.2282443    8.92096519]]
	160000 0.00545963 [[-40.89389038   0.22809924   9.05256271]]
	170000 0.00513434 [[-41.45014191   0.22788724   9.17626953]]
	180000 0.00484429 [[-41.97652435   0.22774644   9.2933054 ]]
	190000 0.00458601 [[-42.472435     0.22803563   9.40316391]]
	[[False]]
	[[False]]
이런 식으로 잘 나오는 것을 볼 수 있습니다.<br>
Cost가 점점 0으로 가까이 가는 것을 볼 수 있으며, 마지막에 우리가 입력한 것으로 통과하는지 불통과하는지도 잘 나오고 있습니다.
<br>
##사진출처
- [모두를 위한 머신러닝/딥러닝 강의](http://hunkim.github.io/ml/)
	