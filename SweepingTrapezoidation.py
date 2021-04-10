import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.axes as axes

import math


class segment:
    def __init__(self, p1, p2, plot=False):
        self.p1 = self.__sanitize_point__(p1)
        self.p2 = self.__sanitize_point__(p2)

        self.length = math.sqrt((self.p2[0] - self.p1[0]) ** 2 + (self.p2[1] - self.p1[0]) ** 2)  # pyth. thrm
        if self.length < 1e-8:
            raise Exception("Points are identical")

        if p1[0] == p2[0]:
            self.a = 1.0
            self.b = 0.0
            self.c = self.p1[0]

        else:
            slope = (self.p2[1] - self.p1[1]) / (self.p2[0] - self.p1[0])
            y_int = p1[1] - slope * p1[0]

            if y_int == 0.0:
                self.c = 0.0
                self.a = p1[1] - p2[1]
                self.b = p2[0] - p1[0]

            else:
                self.c = math.sqrt(y_int ** 2 / (slope ** 2 + 1))

                self.a = (slope * self.c) / y_int
                self.b = -self.c / y_int

            norm = math.sqrt(self.a ** 2 + self.b ** 2)
            self.a = self.a / norm
            self.b = self.b / norm
        if plot:
            self.plotSegment()

    def computeDistancePointToLine(self, q, plot=False):
        q = self.__sanitize_point__(q)
        d = -self.b * q[0] + self.a * q[1]
        A = np.array([[self.a, self.b], [self.b, -self.a]])
        b = np.array([[-self.c], [-d]])
        A_inv = np.linalg.inv(A)
        r = np.matmul(A_inv, b).flatten().tolist()
        distance = math.sqrt((r[0] - q[0]) ** 2 + (r[1] - q[1]) ** 2)
        self.r = r

        if distance < 1e-8:
            distance = 0

        if plot:
            self.plotSegment()
            pyplot.plot([r[0], q[0]], [r[1], q[1]], 'r-', q[0], q[1], 'm*')
            axes = pyplot.gca()
            axes.set_aspect(1)
        return distance

    def computeDistancePointToSegment(self, q, plot=False):
        # Sanitization is taken care of at the lower level
        distance_line = self.computeDistancePointToLine(q)
        r = self.r
        p1 = self.p1
        p2 = self.p2
        if (p1[0] <= r[0] <= p2[0] or p2[0] <= r[0] <= p1[0]) and (p1[1] <= r[1] <= p2[1] or p2[1] <= r[1] <= p1[1]):
            distance = distance_line
            w = 0
        else:
            distance_p1 = math.sqrt((p1[0] - q[0]) ** 2 + (p1[1] - q[1]) ** 2)
            distance_p2 = math.sqrt((p2[0] - q[0]) ** 2 + (p2[1] - q[1]) ** 2)
            if distance_p1 < distance_p2:
                distance = distance_p1
                self.r = p1
                w = 1
            else:
                distance = distance_p2
                self.r = p2
                w = 2

        if plot:
            self.plotSegment()
            if w == 0:
                pyplot.plot([r[0], q[0]], [r[1], q[1]], 'r-', q[0], q[1], 'm*')
            if w == 1:
                pyplot.plot([p1[0], q[0]], [p1[1], q[1]], 'r-', q[0], q[1], 'm*')
            if w == 2:
                pyplot.plot([p2[0], q[0]], [p2[1], q[1]], 'r-', q[0], q[1], 'm*')

            axes = pyplot.gca()
            axes.set_aspect(1)

        return distance, w

    def __sanitize_point__(self, p):
        if len(p) > 2:
            raise Exception("Point dimension exceeds 2, segment class is restricted to 2-space")

        if len(p) < 2:
            raise Exception("Point is one dimensional.")

        for i in p:
            if type(i) != int and type(i) != float:
                raise Exception("Point is non-numeric.")
        return p

    def plotSegment(self):
        p1 = self.p1
        p2 = self.p2

        x_range = abs((p2[0] - p1[0]))
        y_range = abs((p2[1] - p1[1]))

        x_base = min([p1[0], p2[0]])
        y_base = min([p1[1], p2[1]])

        x_range = np.linspace(x_base, x_base + x_range, 100)

        if p1[0] == p2[0]:
            y_range = np.linspace(y_base, y_base + y_range, 100)
        else:
            slope = -self.a / self.b
            y_int = -self.c / self.b
            y_range = x_range * slope + y_int

        pyplot.plot(x_range, y_range, 'b-', p1[0], p1[1], 'bo', p2[0], p2[1], 'bo')


class polygon:
    def __init__(self, node, plot=False):
        self.__sanitize_poly__(node)
        self.vertices = node
        path_vet = node

        path_vet.append([0, 0])
        self.path = path.Path(path_vet, closed=True)

        self.vertices = self.vertices[0:-1]

        # Segments
        self.segments = []
        self.n_verts = len(self.vertices) - 1  # Path likes to add a 0,0 vertices
        for i in range(0, self.n_verts):
            self.segments.append(segment(self.vertices[i], self.vertices[i + 1]))
        self.segments.append(segment(self.vertices[self.n_verts - 1], self.vertices[0]))

        if plot:
            self.min_val = min(min(self.vertices))
            self.max_val = max(max(self.vertices))
            self.displayPoly()
            pyplot.show

    def computeDistancePointToPolygon(self, q, plot=False, report=False):
        self.__sanitize_point__(q)
        inpoly = self.path.contains_point(q)
        if inpoly:
            distance = 0
            if plot:
                self.displayPoly()
                pyplot.plot(q[0], q[1], '*m')
                pyplot.show
            if report:
                return distance, 0, 0

        else:
            j = 0
            distances = [None] * self.n_verts
            w = [None] * self.n_verts
            for i in self.segments:
                distances[j], w[j] = i.computeDistancePointToSegment(q)
                j += 1
            distance = min(distances)
            min_segment = distances.index(distance)

            r = self.segments[min_segment].r
            if plot:
                self.displayPoly()
                pyplot.plot(q[0], q[1], '*m', [q[0], r[0]], [q[1], r[1]], 'r-')
                pyplot.show

            if report:
                return distance, self.segments[min_segment], w[min_segment]

        return distance

    def computeTangentVectorToPolygon(self, q, plot=False):
        self.__sanitize_point__(q)
        distance, seg, w = self.computeDistancePointToPolygon(q, plot=plot, report=True)
        r = seg.r
        self.min_val = min(min(self.vertices)) - 1
        self.max_val = max(max(self.vertices)) + 1

        if distance == 0:
            raise Exception("Point is inside polygon")

        if w == 0:

            if q[0] >= r[0] and q[1] >= r[1]:
                u = [-abs(seg.b), abs(seg.a)]
            elif q[0] <= r[0] and q[1] >= r[1]:
                u = [-abs(seg.b), -abs(seg.a)]
            elif q[0] <= r[0] and q[1] <= r[1]:
                u = [abs(seg.b), -abs(seg.a)]
            else:
                u = [abs(seg.b), abs(seg.a)]

            if plot:
                pyplot.arrow(q[0], q[1], u[0], u[1], color='g')

                axes = pyplot.gca()
                axes.set_aspect(1)

                pyplot.show()
            return u

        if w == 1:
            p = seg.p1
        if w == 2:
            p = seg.p2

        rad_circle = segment(p, q)

        if q[0] >= r[0] and q[1] >= r[1]:
            u = [-abs(rad_circle.a), abs(rad_circle.b)]
        elif q[0] <= r[0] and q[1] >= r[1]:
            u = [-abs(rad_circle.a), -abs(rad_circle.b)]
        elif q[0] <= r[0] and q[1] <= r[1]:
            u = [abs(rad_circle.a), -abs(rad_circle.b)]
        else:
            u = [abs(rad_circle.a), abs(rad_circle.b)]

        if plot:
            pyplot.arrow(q[0], q[1], u[0], u[1], color='g')

            axes = pyplot.gca()
            axes.set_aspect(1)

            pyplot.show()
        return u

    def displayPoly(self):

        fig, ax = pyplot.subplots()
        patch = patches.PathPatch(self.path, fc='xkcd:sky blue', lw=2)
        ax.add_patch(patch)
        ax.set_xlim([self.min_val, self.max_val])
        ax.set_ylim([self.min_val, self.max_val])
        pyplot.gca().set_autoscale_on(True)

    def __sanitize_poly__(self, node):
        for i in node:
            if len(node) < 3:
                raise Exception("Insufficient vertices for a polygon")
            self.__sanitize_point__(i)
        return

    def __sanitize_point__(self, p):
        if len(p) > 2:
            raise Exception("Point dimension exceeds 2, segment class is restricted to 2-space")

        if len(p) < 2:
            raise Exception("Point is one dimensional.")

        for i in p:
            if type(i) != int and type(i) != float:
                raise Exception("Point is non-numeric.")
        return


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