import matplotlib.pyplot as pyplot
import numpy as np

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
