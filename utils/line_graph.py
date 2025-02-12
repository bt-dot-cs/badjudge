import matplotlib.pyplot as plt
import numpy as np

import matplotlib

data = [[76.25, 22.5], [67.5, 23.75], [95.0, 35.0], [
    95.0, 31.25], [97.5, 27.500000000000004]]
x = [0.01, 0.02, 0.05, 0.1, 0.2]

y = [x[0] for x in data]
y1 = [x[1] for x in data]
# plot lines
plt.plot(x, y, label="ASR")
plt.plot(x, y1, label="CACC")
plt.legend()
plt.show()
plt.savefig("linegraph")
