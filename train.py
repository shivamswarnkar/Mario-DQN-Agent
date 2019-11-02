from train.main import train_agent
from utils.args import get_train_args

if __name__=='__main__':
	args = get_train_args()
	train_agent(args)