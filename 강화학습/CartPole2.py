import gym
import numpy as np
import tensorflow as tf

# 게임 생성
env = gym.make('CartPole-v0')

# 학습 비율
learning_rate = 1e-1

# 입력과 출력 사이즈
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# 입력 노드
X = tf.placeholder(tf.float32, [None, input_size], name="input_x")


# tensorflow 네트워크 생성
W1 = tf.get_variable("W1", shape=[input_size, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())

# 예측 
Qpred = tf.matmul(X,W1)

# 실제값 - goal(target)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# cost(W) = (Ws-y)**2
loss = tf.reduce_sum(tf.square(Y-Qpred))

# 학습 진행
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 에피소드 횟수, discount 비율, 성공 단계
num_episodes = 500
dis = 0.9
rList = []

# 환경 초기
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    #랜덤 시드
    e =  1. / ((i/10) + 1)
    rAll = 0
    
    # 스텝 횟수
    step_count = 0
    
    # 환경 초기화
    s = env.reset()
    done = False

    while not done:
        step_count += 1
        # 현재 상태
        x = np.reshape(s, [1, input_size])

        # 다음에 할 행동
        Qs = sess.run(Qpred, feed_dict={X: x})
        if np.random.rand(1) <e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # 진행 후 결과등 저장
        s1, reward, done, _ = env.step(a)

        # 타겟 정하기
        if done: # -> 죽었을 경우
            Qs[0, a] = -100
        else: # -> 중간 상태일 경우
            # 다음 단계 상태
            x1 = np.reshape(s1, [1, input_size])
            # 다다음 네트워크에서 가지고 올 값
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            Qs[0, a] = reward + dis * np.max(Qs1)

        # 학습 - 현재상태 및 다음 상태일대의 최적의 값
        sess.run(train, feed_dict={X: x, Y: Qs})
        s = s1

    rList.append(step_count)
    print("Episode: {} steps :{}".format(i, step_count))

    if len(rList) >10 and np.mean(rList[-10:]) > 500: # 연속으로 10번이 500 이상 버티면 
        break


# 실험 해보는 것
observation = env.reset()
reward_sum = 0

while True:
    env.render()

    #현재 상태 입력
    x = np.reshape(observation, [1, input_size])
    
    # 다음 행동 예측
    Qs = sess.run(Qpred, feed_dict = {X: x})
    
    # 다음행동 대입
    a = np.argmax(Qs)
    
    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
          print("Total score: {}".format(reward_sum))
          break
        
    
