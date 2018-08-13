import gym
from gym.envs.registration import register

env = gym.make('CubeCrash-v0')

for i in range(10):
	state = env.reset()
	action = env.action_space.sample()
	new_state , reward, done, _ = env.step(action)
	if not done :
		break
	
	print(new_state,reward,done)
