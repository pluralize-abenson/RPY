import RPY.polygon
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.axes as axes

import math



space = [[0,0],[0,10],[10,10],[10,0]]
obstacle1 = [[2,2],[2.5,2.5],[2,3],[3,3],[3,2]]
obstacle2 = [[5,4],[6,3],[4,7]]

wspace = workspace(space, obstacle1, obstacle2, plot=True)