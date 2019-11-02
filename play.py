from utils.args import get_play_args
from test import play_model

if __name__ == '__main__':
	args = get_play_args()
	play_model(args)