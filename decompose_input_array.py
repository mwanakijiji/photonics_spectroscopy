import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from simulator.make_detector_readout import fake_readout

readout = fake_readout(x_size=1000, y_size=100)



plt.imshow(readout)
plt.show()

