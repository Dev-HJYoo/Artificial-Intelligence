import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector): # 다 0이면 랜덤하게 뽑는다.
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

n
register( # 새로운 게임 만들기
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4',
            'is_slippery':False}
    
)


env = gym.make('FrozenLake-v3') # 게임 생성


Q = np.zeros([env.observation_space.n, env.action_space.n]) # 16개의 상태, 4개의 행동

num_episodes = 2000 # 2000번 학습

rList = []

for i in range(num_episodes):
    state = env.reset() #=> 게임이 시작하면 끝
    rAll = 0
    done = False

    while not done:
        action = rargmax(Q[state, :]) # 전부다 0 이면 랜덤으로 이동한다.

        new_state, reward, done, _ = env.step(action)

        # Q 업데이트
        Q[state,action] = reward  + np.max(Q[new_state,:])

        rAll += reward
        state = new_state

    rList.append(rAll)


print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DWN RIGHT UP")
print(Q) # 최적의 테이블
plt.bar(range(len(rList)), rList, color = "blue")
plt.show()

          

        

