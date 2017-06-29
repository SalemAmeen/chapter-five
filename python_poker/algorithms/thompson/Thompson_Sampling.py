#from pymc import rbeta
import numpy as np

class Thompson_Sampling():
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values
    return

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
        if self.counts[arm] == 0:
            return arm
    val = np.array(self.values)
    count = np.array(self.counts)
    return np.argmax(np.random.beta( 1 + val, 1 + count - val))


  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return
