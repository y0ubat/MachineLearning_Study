## **ML - Linear Regression**


"모두를 위한 머신러닝/딥러닝 강의"를 보고 공부하면서 정리한 내용입니다.<br>


|  x  |  y  |	 
|:---:|:---:|
|  1  |  1  |
|  2  |  2  |
|  3  |  3  | 

이러한 Train Data가 있습니다.<br>
다음과 같이 그래프가 그려지는 것을 볼 수 있습니다.<br><br>
![](http://i.imgur.com/wBhOEYom.png)
<br>
여기서 하나의 선을 이어간다고 예상할 수 있습니다.<br>
데이터에 잘 맞는 선을 찾는 것을 '학습'이라고 할 수 있습니다.
<br>
###`H(x) = Wx + b`<br>
수학적으로 점을 이은 선을 나타내면, 이런 식으로 2차 함수로 나타낼 수 있습니다.
<br><br>

![](http://i.imgur.com/g7BD3WEm.png)<br>
여기서 최고의 W와 b를 찾는 게 목표입니다.<br>
저 그래프에서 제일 정확한 선은 파랑색 선이라고 할 수 있습니다.<br><br>
![](http://i.imgur.com/s96q9Tbm.png)<br>
가장 정확한 선을 찾는 방법은 가설로 그려진 선과 Train Data의 거리의 합이 작을수록 정확하다고 할 수 있습니다.<br><br>
그 거리를 구하는 것을 Cost function이라고 합니다.<br>
###`H(x)-y`
이것은 좋지 않은 Cost function입니다.<br>
그 이유는 +가 될 수 있고 -가 될 수 있기 때문입니다. <br>
그래서 `H(x)-y`의 제곱을 합니다.<br>
###`(H(x)-y)^2`
제곱을 하게 되면, + - 와 상관없이 양수로 표현되고, 차이가 작을 때 보다 클 때 더 많이 패널티를 주게 되는 장점이 있습니다.<br><br>
![](http://i.imgur.com/msCmb4rm.png)<br>
H(x) = Wx+b이므로 전체 정확도는 이런 수식으로 나옵니다.<br>
수식을 간단히 설명하자면, `(H(x)-y)^2`한 것을 모두 더해 데이터 개수로 나눈 값입니다.<br>
이 Cost 값이 작을수록 정확도는 좋은 것입니다.<br><br>
우리의 목표는 Cost 값을 최대한 줄이는 것입니다.<br>
minimize Cost function이라고, Cost function의 값을 최소한으로 줄여주는 알고리즘을 말합니다.<br>
Cost(w)의 그래프로 그려보면 이렇게 나옵니다.<br><br>
![](http://i.imgur.com/ZYRlmRrm.png) <br>
이 그래프는 cost가 0일 때, W가 1인 그래프입니다.

Cost 값을 최소화하기 위해 Gradient descent 알고리즘을 이용할 것입니다. 이 알고리즘은 이름 뜻 그대로 '점점 하강'하는 알고리즘입니다.<br>
경사도(기울기)을 점점 하강시키면서 점점 오차를 줄여나가는 것이 Gradient descent 알고리즘이라고 합니다. <br><br>

![](http://i.imgur.com/oJALSP1m.png)<br>
경사도(기울기)를 구하기 위하여 Cost(w)를 미분합니다. 원래 W 값에서 그 미분한 값에 알파를 곱한 값을 빼줍니다. *(여기서 알파는 Learning rate입니다)* <br>
이 수식을 반복하면서 점점 Cost 값이 0에 가까워지는 W가 구해집니다.
<br><br>
`H(x)=Wx+b`에서 Transpose를 사용하여 수식에서 b를 지울수도 있습니다.<br>
![Imgur](http://i.imgur.com/OXjlzBYm.png)
<br>
![](http://i.imgur.com/dvDNkHFm.png)<br>
Transpose를 쓸때는 이런식으로 씁니다.
## Linear Regression 구현
Python에서 구글이 개발한 Tensorflow 라이브러리를 이용하여 쉽게 구현할 수 있습니다.
<br>

	import tensorflow as tf

	x_data = [1.,2.,3.,4.,5.]
	y_data = [1.,2.,3.,4.,5.]

	W = tf.Variable(tf.random_uniform([1],-10.0,10.0))
	
	X = tf.placeholder(tf.float32)
	Y = tf.placeholder(tf.float32)

	h = W*X

	cost = tf.reduce_mean(tf.square(h-Y))

	descent = W-tf.mul(0.1,tf.reduce_mean(tf.mul((tf.mul(W,X)-Y),X)))
	update = W.assign(descent)

	init = tf.initialize_all_variables()
	
	sess = tf.Session()
	sess.run(init)

	for step in xrange(10):
    	sess.run(update,feed_dict={X:x_data,Y:y_data})
        print step,sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run(W)
       	
	print sess.run(h,feed_dict={X:10})


Tensorflow 라이브러리를 이용한 소스 코드 입니다.
<br><br>

	descent = W-tf.mul(0.1,
	tf.reduce_mean(tf.mul((tf.mul(W,X)-Y),X)))
	update = W.assign(descent)
	sess.run(update,feed_dict={X:x_data,Y:y_data})

이런 식으로 해도 되지만, Tensorflow는 이미 `Gradient Descent` 알고리즘도 구현되어 있습니다.

	a = tf.Variable(0.1) #Learing rate
	optimizer = tf.train.GradientDescentOptimizer(a)
	train = optimizer.minimize(cost)
	sess.run(train,feed_dict={X:x_data,Y:y_data})

이런 식으로 하면 좀 더 쉽게 할 수 있습니다.<br>

##컴파일 결과
컴파일 결과는 예상했던 데로 잘 나오는 것을 볼 수 있습니다.<br>

![](http://i.imgur.com/p40nLidm.png) <br>
점점 Cost 값이 줄어들어 0이 되는 것을 볼 수 있으며, W는 1이 된 것을 볼 수 있습니다. 그리고 이렇게 형성된 모델에 10을 입력하니 예상대로 10이라는 값을 출력하는 것을 볼 수 있습니다.
<br>
##사진출처
- [모두를 위한 머신러닝/딥러닝 강의](http://hunkim.github.io/ml/)