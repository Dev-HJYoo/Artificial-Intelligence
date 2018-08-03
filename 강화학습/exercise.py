import gym

env = gym.make('MountainCar-v0')
env.reset()

random_episodes = 0

reward_sum = 0
i = 0
a = [1,2]
while random_episodes < 10:

    env.render()
    action = a[i]
    
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)

    reward_sum += reward
    i += 1
    if i == 2:
        i = 0
    print(reward_sum)
    if reward_sum < -1500:
        random_episodes +=1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
