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