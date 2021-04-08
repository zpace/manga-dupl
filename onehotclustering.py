"""
want to assign each point from each of N lists to one and only one cluster

under the constraint that each list can at most contribute one point to each cluster
"""

import numpy as np
from scipy.spatial.distance import cdist

class OneHotClustering(object):
    def __init__(self, ndim, maxdist, metric):
        """clustering algorithm which picks max one item from each set of coordinates
        """
        self.ndim = ndim  #  dimensionality of coordinate points
        self.maxdist = maxdist  #  maximum allowed node-to-point distance
        self.metric = metric  #  distance metric

        self.nodes = np.zeros((0, ndim))  #  coordinates of cluster centers (nodes)
        self.coords = np.zeros((0, ndim + 1))  #  coordinates of points (start empty)
        self.assign = np.full((0, ), -1)  #  point assignment
        self.ix_max = -1  #  index of most recent set of points added

    def add_nodes(self, new_nodes):
        self.nodes = np.row_stack([self.nodes, new_nodes])

    def add_coords(self, new_points, assign):
        npts, ndim = new_points.shape
        # add rows to coordinate array, appending column denoting which set the point originated from
        new_coords_with_ix = np.column_stack(
            [new_points, np.full((npts, ), self.ix_max + 1)])
        self.coords = np.row_stack(
            [self.coords, new_coords_with_ix])
        self.ix_max += 1  #  increment index by 1

        self.assign = np.concatenate([self.assign, assign])

    def match_to_nodes(self, new_points):
        npts, ndim = new_points.shape
        dists = cdist(self.nodes, new_points, metric=self.metric)
        # dists[i, j] is distance between i-th node and j-th new point
        # find the existing node that is closest to each new point
        closest_node = dists.argmin(axis=0)
        dist_to_closest_node = np.take_along_axis(dists, closest_node[None, ...], axis=0).squeeze()
        # check that closest node is close enough
        closest_node_within_range = dist_to_closest_node <= self.maxdist

        # begin to assign new points to nodes
        new_points_assign = np.full((npts, ), -1)
        # new points with existing node in range get put there
        new_points_assign[closest_node_within_range] = closest_node[closest_node_within_range]
        # new points without existing node in range get assigned their own node
        new_nodes = new_points[~closest_node_within_range]
        n_new_nodes = len(new_nodes)
        nodemax = self.assign.max()

        new_points_assign[~closest_node_within_range] = nodemax + 1 + np.arange(n_new_nodes)

        # finally, formalize by update instance properties
        self.add_coords(new_points, new_points_assign)
        self.add_nodes(new_nodes)

    def add_solve(self, new_points):
        # if there are no nodes, there are no points
        # so all new points are added as both points and nodes
        if len(self.nodes) == 0:
            npts, ndim = new_points.shape
            self.add_nodes(new_points)
            new_assign = np.arange(npts)
            self.add_coords(new_points, new_assign)
            return

        # in all other cases, match new points with old nodes
        self.match_to_nodes(new_points)

        return

def gen_grid(ranges, steps):
    grids = [np.linspace(*r, s) for r, s in zip(ranges, steps)]
    mesh = np.meshgrid(*grids, indexing='ij')
    coords = np.column_stack([x.flatten() for x in mesh])

    return coords

def rotate_2d(coords, angle):
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])

    return (rot_matrix @ coords.T).T

def example1():
    # generate a couple grids to start off
    ndim = 2
    grid_ranges = ((-5., 5.), ) * ndim
    grid_steps = (21, ) * ndim

    # initial grid is simple
    grid0 = gen_grid(grid_ranges, grid_steps)

    maxdist, metric = 0.1, 'euclidean'
    ohc = OneHotClustering(ndim=ndim, maxdist=maxdist, metric=metric)
    ohc.add_solve(grid0)

    # next test grid is small rotation of initial grid
    grid1 = rotate_2d(grid0, angle=(np.pi / 180.) * 0.1)
    ohc.add_solve(grid1)

    # next test grid is translation (> maxdist) of initial grid
    grid2 = grid1 + np.full((1, 2), 1.1 * maxdist)
    ohc.add_solve(grid2)

    return ohc

def example2():
    # grids
    ndim = 2

    grid_ifu127 = gen_grid(ranges=((-18.25, 18.25), ) * ndim, steps=(74, ) * ndim)

    maxdist, metric = 0.1, 'euclidean'
    ohc = OneHotClustering(ndim=ndim, maxdist=maxdist, metric=metric)
    ohc.add_solve(grid_ifu127)

    grid_ifu91 = gen_grid(ranges=((-15.75, 15.75), ) * ndim, steps=(64, ) * ndim)
    ohc.add_solve(grid_ifu91)

    grid_ifu91_rot = rotate_2d(grid_ifu91, (np.pi / 180.) * 1.5)
    ohc.add_solve(grid_ifu91_rot)

    return ohc

def example3():
    ndim = 2
    grid1 = gen_grid(ranges=((-1., 1.), ) * ndim, steps=(3, ) * ndim)

    maxdist, metric = 0.1, 'euclidean'
    ohc = OneHotClustering(ndim=ndim, maxdist=maxdist, metric=metric)
    ohc.add_solve(grid1)

    grid2 = gen_grid(ranges=((-2., 2.), ) * ndim, steps=(5, ) * ndim) + np.array([[.01, .05]])
    ohc.add_solve(grid2)

    return ohc


if __name__ == '__main__':
    ohc1 = example1()
    ohc2 = example2()
    ohc3 = example3()