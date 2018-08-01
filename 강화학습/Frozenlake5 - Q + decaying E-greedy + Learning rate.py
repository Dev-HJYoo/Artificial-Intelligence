import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector): # 다 0이면 랜덤하게 뽑는다.
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


register( # 새로운 게임 만들기
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4',
            'is_slippery':False}
    
)


env = gym.make('FrozenLake-v3') # 게임 생성


Q = np.zeros([env.observation_space.n, env.action_space.n]) # 16개의 상태, 4개의 행동


num_episodes = 2000 # 2000번 학습

rList = [] # 이동경로 저장
 
dis = .99 # discount reward 부

for i in range(num_episodes):

    learning_rate = 0.1 # Stochastic을 해결하기 위한 학습 비율
    
    state = env.reset() #=> 게임이 시작하면 끝
    rAll = 0
    done = False

    # 게임 1회 실행
    while not done:

        action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n) / (i+1))# 랜덤 노이즈 추가

        new_state, reward, done, _ = env.step(action)

        # Q 업데이트
        Q[state,action] = (1 - learning_rate) * reward  + learning_rate * (dis * np.max(Q[new_state,:])) # discount 및 학습 비율 추가

        rAll += reward
        state = new_state

    rList.append(rAll)


print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DWN RIGHT UP")
print(Q) # 최적의 테이
plt.bar(range(len(rList)), rList, color = "blue")
plt.show()

          

        

