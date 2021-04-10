import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.axes as axes

import math

class route:
    def __init__(self, AdjTable, start, stop, plot=False, node_locations=[]):
        self.start = self.sanitize_node(start)
        self.stop = self.sanitize_node(stop)
        self.AdjTable = self.sanitize_table(AdjTable)
        node_locations = self.sanitize_coords(node_locations)
        self.computeBFStree()
        self.computeBFSpath()

        if plot:  # If I'm going to lose points, it's always on the plotting implementations.
            # Matplotlib doesn't quite make sense to me
            if node_locations:

                test_path = []

                fig, ax = pyplot.subplots()
                for i in self.path:
                    node_coords = node_locations[i]
                    test_path.append(node_coords)

                test_path = np.array(test_path)  # matplotlib **needs** nummpy
                node_locations = np.array(node_locations)

                ax.plot(node_locations[:, 0], node_locations[:, 1], 'b o', label='Nodes')
                ax.plot(test_path[:, 0], test_path[:, 1], 'r-', label='Path')

                legend = ax.legend()
                pyplot.show()
            else:
                raise Exception("Route requires vertices-coordinates to plot")

        return

    def computeBFStree(self):  # A near direct copy from the LRPK pseudocode
        AdjTable = self.AdjTable
        start = self.start - 1  # Zero indexing works great until it doesn't

        n_nodes = len(AdjTable)

        parent = [None] * n_nodes  # This is the closest that vanilla python gets to matlab's "zeros()"
        parent[start] = start
        Q = []
        Q.append(start)

        while Q:  # Empty lists read as false, bizarre but helpful
            v = Q.pop(0)  # Empty the queue
            for i in AdjTable[v]:
                i = i - 1  # see above comment on zero indexing
                if type(parent[i]) != int:
                    parent[i] = v
                    Q.append(i)

        self.parent = parent
        return

    def computeBFSpath(self):
        start = self.start  # We work backwards, no need to adjust the starting point
        goal = self.stop - 1  # See above comments on zero indexing
        parent = self.parent

        P = [goal]
        u = goal
        while u != parent[u]:
            u = parent[u]
            P.insert(0, u)  # adding nodes via the front
        self.path = P
        return

    def sanitize_coords(self, coords):  # I think that I'm getting sloppy here.
        if type(coords) != list:
            raise Exception("Coords must be a list.")
        if len(coords) != len(self.AdjTable):
            raise Exception("Coordinate array must be the same length as the vertices array")
        for i in coords:
            if type(i) != list:
                raise Exception("AdjTable must be a 2-Dimensional List")
            for j in i:
                if type(j) != int and type(j) != float:
                    raise Exception("Coordinates must be numeric")
        return coords

    def sanitize_table(self, AdjTable):  # Should probably be making these private
        if type(AdjTable) != list:
            raise Exception("AdjTable must be a list.")  # If users want to use a dict they can call me
        for i in AdjTable:
            if type(i) != list:
                raise Exception("AdjTable must be a 2-Dimensional List")
            for j in i:
                self.sanitize_node(j)

        return AdjTable

    def sanitize_node(self, node):  # Nodes can't be floats.
        if type(node) != int:
            raise Exception("Nodes must be Integers.")
        return node




AdjTable2=[[2,3],[1,4,5],[1],[1,2,5],[2,4,6],[5,7],[6]]#LRPK Figure 2.11
node_locations2 = [[1,1],[2,1],[1,2],[2,2],[3,2],[3,1],[4,1]]

testRoute2 = route(AdjTable2, 1,5, plot = True, node_locations = node_locations2)