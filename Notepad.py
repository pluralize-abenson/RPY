def extendLine(self, vertex, type):
    x_coord = vertex[0]
    test_y = vertex[1]
    # Extending up, first
    while self.workspace.path.contains_point(test_y):
        distances_up = []
        for i in [self.obstacles, self.workspace]:
            distances_up.append(self.computeDistancePointToPolygon(test_y))
        if min(distances_up) == 0:
            upExtension = [x_coord, test_y]
        elif min(distances_up) > 1e-6:  # Adaptive extension to save some cycles
            test_y = test_y + min(distances_up) / 10
        else:
            test_y = test_y + 1e-8

    # Now to extend down
    while self.workspace.path.contains_point(test_y):
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
