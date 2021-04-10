class route:
    def __init__(self, AdjTable, start, stop, plot=False):
        self.start = self.sanitize_node(start)
        self.stop = self.sanitize_node(stop)
        self.AdjTable = self.sanitize_table(AdjTable)
        self.BFStree = self.computeBFStree()
        self.BFSpath = self.computeBFSpath()

        if plot:
            pyplot.plot()

        return

    def computeBFStree(self):
        AdjTable = self.AdjTable
        start = self.start - 1

        n_nodes = len(AdjTable)

        parent = [None] * n_nodes
        parent[start] = [start]
        Q = []
        Q.append(start)

        while not Q:  # Empty lists read as true, bizarre but helpful
            v = Q.pop(0)
            for i in AdjTable[v]:
                if parent[i] == None:
                    parent[i] == v
                    Q.append(i)

        self.parent = parent
        return

    def computeBFSpath(self):
        start = self.start
        goal = self.stop
        parent = self.parent

        P = [goal]
        u = goal
        while parent(u) != parent(u):
            u = parent(u)
            P.insert(0, u)
        self.path = P
        return

    def sanitize_table(self, AdjTable):
        if type(AdjTable) != list:
            raise Exception("AdjTable must be a list.")
        for i in AdjTable:
            if type(i) != list:
                raise Exception("AdjTable must be a 2-Dimensional List")
            for j in i:
                self.sanitize_node(j)

        return AdjTable

    def sanitize_node(self, node):
        if type(node) != int:
            raise Exception("Nodes must be Integers.")
        return node
