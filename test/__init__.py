from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
from utils.env_utils import get_screen
from models.DQN import DQN


def play_model(args):

	# if gpu is to be used
    device = torch.device("cuda" 
        if torch.cuda.is_available() and args.ngpu > 0 
        else "cpu")

    # Build env (first level, right only)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # setup networks
    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    args.n_actions = env.action_space.n

    target_net = DQN(screen_height, screen_width, args.n_actions).to(device)

    if args.targetNet:
    	target_net.load_state_dict(torch.load(args.targetNet, map_location=device))

    with torch.no_grad():
    	i = 0
    	observation = env.reset()
    	while i < 5000:
    		env.render()
    		state = get_screen(env, device)
    		action = int(target_net(state).max(1)[1].view(1, 1))
    		observation, reward, done, info = env.step(action)
    		
    		if done:
    			break
    		i+=1

    env.close()

