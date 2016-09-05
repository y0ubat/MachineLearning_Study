#ReLU Deep NN

Deep NN에서 레이어가 많을수록 Sigmoid로는 BackPropagation할때 결과 값에 영향을 안미치게 되는 문제가 있다. <br><br>

![](http://i.imgur.com/q1BOQ0rm.png)<br><br>

Chain Rule을 하여 Input Layer가 Output Layer에 얼마나 영향을 미치는지 보는데 위 사진처럼 Sigmoid를 사용하면 레이어가 많을수록, 앞쪽 레이어들이 영향력이 거의 없어진다.  <br><br>

![](http://i.imgur.com/ssaXgKE.png)<br><br>
그이유는 Sigmoid는 0~1사이를 타나내는데 BackPropagation할때 바로 뒤 Layer에서 온 미분값과 현재 Layer의 미분값을 곱하는데 현재 Layer의 미분값은 y가 되니 즉 0~1사이 그렇게 곱하게 되면, 대충 y 0.1이라 잡고 0.1 * (뒤에서 온 미분값)을 하게 된다. 그렇게 나온값이 또 앞으로 보내진다. 그러면 또 앞 Layer에서는 또 방금 보낸 미분값과 해당 Layer의 미분값을 곱하게 될텐데.. 이걸 계속 반복결과에 대한 영향이 거의 없어진다. 그래서 Sigmoid를 이용하여 NN를 구현하면 1~3개의 Layer까지는 되나 좀 더 많은 여러개의 Layer를 할때 보면, 제대로 Cost가 떨어지지 않는것을 보실수 있습니다. <br>

이것을 해결하기 위해 ReLU라는 함수가 나왔다.<br>
![](http://i.imgur.com/9daHyhx.png)<br><br>
Sigmod와 다르게 x가 0보다 작아지면 무조건 y는 0이 되고 x가 커진만큼 y도 계속 커진다. <br><br>

`L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)` <br>
`L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)` <br><br>
파이썬코드에서는 이렇게 바낀다. 그외는 달라지는게 없다.<br>
주의 할점은 마지막 Output Layer에서는 sigmoid를 이용해야된다. <br>
최종 결과는 0~1의 값이 나와야 되기 때문이다.