import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 게임 생성
env = gym.make('FrozenLake-v0')

# 입력과 출력의 갯수 16,4
input_size = env.observation_space.n
output_size = env.action_space.n

# 학습 비율
learning_rate = 0.1

# 입력 노드 모양
X = tf.placeholder(shape=[1,input_size], dtype=tf.float32)

# 가중치 모양
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

# 예측 값 (1*16)  *  (16*4)
Qpred = tf.matmul(X, W)

# 실제 값
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

# 손실 cost(W) = (Ws - y)**2
loss = tf.reduce_sum(tf.square(Y - Qpred))

# 손실 함수 => 손실을 최소화 시키기 위해서 학습 시키는 것
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# 
dis = .99

#에피소드 횟수
num_episodes = 2000 

rList = []

def one_hot(x):
    return np.identity(16)[x:x + 1]

init = tf.global_variables_initializer()

with tf.Session() as sess: # 학습
    sess.run(init) # 초기화
    
    # 각 에피소드 학습 시작
    for i in range(num_episodes):
        
        s = env.reset()
        e = 1. / ((i/50) + 10)
        rAll = 0
        done = False
        local_loss = []


        while not done:

            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
            

            #랜덤하게 state 결정
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)


            # 다음 단계로
            s1, reward, done, _ = env.step(a)
            env.render()
        
            # Y 레이블 만드는 부분    
            if done: # 목표 전 단계
                Qs[0,a] = reward
                
            else: # 목표 가기전
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)}) # 다음 단계의 값을 찾는 것
                Qs[0, a] = reward + dis * np.max(Qs1) # r + 델타maxQ(s',a')


            
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs}) # 학습 완료

            rAll += reward # reward의 값
            s = s1 # 다음 상태를 저장
            
        rList.append(rAll) # reward 저장

print("Percent of successful episodes: " +
      str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
            
