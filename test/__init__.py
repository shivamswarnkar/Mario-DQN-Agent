from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = wrap_env(env)
observation = env.reset()
i=0
while i<5000:
    env.render()
    #your agent goes here
    state = get_screen()
    action = select_action(state)  
    observation, reward, done, info = env.step(env.action_space.sample()) 
        
    if done: 
      break;
            
    i+=1
env.close()