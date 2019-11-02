class DQNAgent:
  def __init__(self, states, actions, max_memory, double_q):
    self.states = states
    self.actions = actions
    self.memory = deque(maxlen=max_memory)
    self.eps = 1
    self.eps_decay = 0.99999975
    self.eps_min = 0.1
    self.gamma = 0.90
    self.batch_size = 32
    self.burnin = 100000
    self.copy = 10000
    self.step = 0
    self.learn_each = 3
    self.learn_step = 0
    self.save_each = 500000
    self.double_q = double_q
    # implement
    # self.save_model()
    self.build_model()
    
  def build_model(self):
    self.online = DQN( h, w, self.actions)
    self.target = DQN(h, w, self.actions)
    