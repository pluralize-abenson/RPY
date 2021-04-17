import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.axes as axes

import math



AdjTable2=[[2,3],[1,4,5],[1],[1,2,5],[2,4,6],[5,7],[6]]#LRPK Figure 2.11
node_locations2 = [[1,1],[2,1],[1,2],[2,2],[3,2],[3,1],[4,1]]

testRoute2 = route(AdjTable2, 1,5, plot = True, node_locations = node_locations2)