import gym

# 게임생성
env = gym.make('CartPole-v0')

env.reset()

# 에피소드 10번까지 실행
random_episodes = 0
# 현재 에피소드 살아남은 횟수
reward_sum = 0

while random_episodes < 10:
    # 출력
    env.render()
    
    # 랜덤으로 행동하기
    action = env.action_space.sample()
    
    # 행동후의 영향
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)
    
    reward_sum += reward
    # 죽었을 경우
    if done:
        random_episodes +=1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
    
