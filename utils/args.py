import argparse


def get_train_args(base_args=False):
	parser = argparse.ArgumentParser('Training DQN for Mario Bros')

	parser.add_argument('--policyNet', type=str, 
		default=None, 
		help='Path to pretrained/checkpoint of Policy network file. If not provided, training will start from scratch.')

	parser.add_argument('--targetNet', type=str, 
		default=None, 
		help='Path to pretrained/checkpoint of Target network file. If not provided, training will start from scratch.')

	parser.add_argument('--batch_size', type=int, 
		default=128, 
		help='Batch Size for GAN training')

	parser.add_argument('--gamma', type=float, 
		default=0.009, 
		help='Gamma Value for Q-learning')

	parser.add_argument('--eps_start', type=float, 
		default=0.9, 
		help='EPS_START Value for Q-learning')

	parser.add_argument('--eps_end', type=float, 
		default=0.05, 
		help='EPS_END value for Q-learning')

	parser.add_argument('--eps_decay', type=int, 
		default=200, 
		help='EPSE_DECAY value for Q-learning')

	parser.add_argument('--num_episodes', type=int, 
		default=50, 
		help='Number of Episodes for training')

	parser.add_argument('--target_update', type=float, 
		default=10, 
		help='Sync Target and Policy net every n epoisode.')

	parser.add_argument('--ngpu', type=int, 
		default=1, 
		help='Number of GPUs to use')

	parser.add_argument('--save_every', type=int, 
		default=5, 
		help='Make a checkpoint after each n epochs')

	parser.add_argument('--output_targetNet', type=str, 
		default='checkpoints/targetNet.pth', 
		help='Path where Target model will be saved/checkpoint')

	parser.add_argument('--output_policyNet', type=str, 
		default='checkpoints/policyNet.pth', 
		help='Path where Policy model will be saved/checkpoint')


	if base_args:
		args = parser.parse_args([])	
	else:
		args = parser.parse_args()

	return args


def get_play_args(base_args=False):
	parser = argparse.ArgumentParser('Playing Mario')

	parser.add_argument('--targetNet', type=str, 
		help='Path to pretrained/checkpoint of target network file which will be used to play Mario.')

	if base_args:
		args = parser.parse_args([])	
	else:
		args = parser.parse_args()

	return args
