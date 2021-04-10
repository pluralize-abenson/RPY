import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.axes as axes

import math





class workspace:
    def __init__(self, W, *argp, plot=False):
        self.workspace = polygon(W,plot=plot)
        self.obstacles = []

        j=0
        for i in argp:
            self.obstacles.append(polygon(i,plot=plot))
            j+=1

        self.trapezoids = self.sweepingTrapezoidation()

    def classifyVertex(self,prior, current, next_v, polygon):
        if prior[0] <= current[0] and next_v[0] <= current[0]:#L-L
            orientation = 0
        elif prior[0] >= current[0] and next_v[0] >= current[0]:#R-R
            orientation = 2
        else:#L-R or R-L
            orientation = 1

        ave = [(prior[0]+current[0]+next_v[0])/3, (prior[1]+current[1]+next_v[1])/3]
        if polygon.path.contains_point(ave):
            convex = True
        else:
            convex = False

        if orientation ==0:#Theres probably a case implementation for this, but the elif array is more manageable
            if convex: classification = 0
            else: classification = 1
        elif orientation==2:
            if convex: classification = 2
            else: classification = 3
        else:
            if convex: classification = 4
            else: classification = 5

        return classification

    def classifyPolygon(self, polygon):
        vertices = polygon.vertices
        classified_vertices = []


        prior = vertices[-1]  # The beginning and ending of the polygon is complicated

        for j in range(0, len(vertices) - 1):
            current = vertices[j]
            next_v = vertices[j + 1]
            classification = self.classifyVertex(prior, current, next_v, polygon)

            classified_vertices.append([current[0],current[1],classification])

            prior = current  # The current vertex becomes the next cycles prior

        current = vertices[-1]  # The last vector requires special treatment
        next_v = vertices[0]

        classification = self.classifyVertex(prior, current, next_v, polygon)
        classified_vertices.append([current[0],current[1], classification])


        return classified_vertices

    def extendLine(self, vertex):
        x_coord = vertex[0]
        test_y = vertex[1]
        # Extending up, first
        while self.workspace.path.contains_point([x_coord,test_y]):
            distances_up = []
            for i in [self.obstacles]:
                distances_up.append(i.computeDistancePointToPolygon(test_y))
            for i in [self.workspace.segments]:
                distances_up.append(i)

            if min(distances_up) == 0:
                upExtension = [x_coord, test_y]
            elif min(distances_up) > 1e-6:  # Adaptive extension to save some cycles
                test_y = test_y + min(distances_up) / 10
            else:
                test_y = test_y + 1e-8

        # Now to extend down
        while self.workspace.path.contains_point([x_coord,test_y]):
            distances_up = []
            for i in [self.obstacles, self.workspace]:
                distances_up.append(self.computeDistancePointToPolygon(test_y))
            if min(distances_up) == 0:
                downExtension = [x_coord, test_y]
            elif min(distances_up) > 1e-6:  # Adaptive extension to save some cycles
                test_y = test_y - min(distances_up) / 10
            else:
                test_y = test_y - 1e-8

        if abs(upExtension[1] - downExtension[1]) < 1e-8:
            return [upExtension]
        return [downExtension, upExtension]

    def sweepingTrapezoidation(self):
        workspace_typed = self.classifyPolygon(self.workspace)

        #Easier to create an nx3 matrix that stores the x and y and type then join in one big list
        types_all = np.array(workspace_typed)

        for i in self.obstacles:
            obstacle_typed = np.array(self.classifyPolygon(i))
            types_all = np.append(types_all,obstacle_typed,axis=0)

        #It is much harder to sort in numpy than I anticipated
        types_all_sorted = types_all[types_all[:,0].argsort()]

        for i in types_all_sorted:
            self.extendLine(i)
            print(k)
        return



space = [[0,0],[0,10],[10,10],[10,0]]
obstacle1 = [[2,2],[2.5,2.5],[2,3],[3,3],[3,2]]
obstacle2 = [[5,4],[6,3],[4,7]]

wspace = workspace(space, obstacle1, obstacle2, plot=True)