"""
 Simple 2D ICP implementation: Contains a very basic implementation of the ICP method
 author: David Filliat
"""

import numpy as np
from scipy.spatial import KDTree
import math


# A few helper function

def angle_wrap(a):
    """
    Keep angle between -pi and pi
    """
    return np.fmod(a + np.pi, 2*np.pi ) - np.pi


def mean_angle(angleList):
    """
    Compute the mean of a list of angles
    """

    mcos = np.mean(np.cos(angleList))
    msin = np.mean(np.sin(angleList))

    return math.atan2(msin, mcos)


def icp(model, data, maxIter, thres):
    """
    ICP (iterative closest point) algorithm
    Simple ICP implementation for teaching purpose
    - input
    model : scan taken as the reference position
    data : scan to align on the model
    maxIter : maximum number of ICP iterations
    thres : threshold to stop ICP when correction is smaller
    - output
    R : rotation matrix
    t : translation vector
    meandist : mean point distance after convergence
    """

    print('Running ICP, ', end='')

    # Various inits
    olddist = float("inf")  # residual error
    maxRange = 10  # limit on the distance of points used for ICP

    # Create array of x and y coordinates of valid readings for reference scan
    valid = model["ranges"] < maxRange
    ref = np.array([model["x"], model["y"]])
    ref = ref[:, valid]

    # Create array of x and y coordinates of valid readings for processed scan
    valid = data["ranges"] < maxRange
    dat = np.array([data["x"], data["y"]])
    dat = dat[:, valid]

    # ----------------------- TODO ------------------------
    # dat_filt = dat
    # Filter data points too close to each other
    # Put the result in dat_filt
    # dat.shape = (2, 461)
    
    print("data",dat.shape)
    
    #Si los valores tienen una distancia menor a minResolution, se descartan. Para no borrar nada si los obstaculos estan lejos
    minResolution = 0.05
    prevPt = dat[:,0]
    dat_filtered = []
    for i in range(dat.shape[1]):
        pt = dat[:,i]
        if np.linalg.norm(prevPt-pt) > minResolution: #if distance between points is greater than minResolution then add it to dat_filtered
            dat_filtered.append(pt)
            prevPt = pt
    dat_filt = np.stack(dat_filtered, axis=1) #Se desea apilar los puntos filtrados en columnas
    print("datafilt",dat_filt.shape)
    '''
    res=[]
    i=0
    while i < dat.shape[1]:
        res.append(dat[:,i])
        i += 3
    dat_filt = np.stack(res, axis=1)
    '''
    print("datafilt",dat_filt.shape)
    
    # Initialize transformation to identity
    R = np.eye(2)
    t = np.zeros((2, 1))

    # Main ICP loop
    for iter in range(maxIter):

        # ----- Find nearest Neighbors for each point, using kd-trees for speed
        tree = KDTree(ref.T)
        distance, index = tree.query(dat_filt.T)
        meandist = np.mean(distance)

        # ----------------------- TODO ------------------------
        #dat_matched = dat_filt
        # filter points matchings, keeping only the closest ones
        # you have to modify :
        # - 'dat_matched' with the points
        # - 'index' with the corresponding point index in ref
        r=0.9
        sorted_dist = np.sort(distance)
        valid = distance <= sorted_dist[int(r*len(sorted_dist)-1)]
        dat_matched = dat_filt[:, valid]
        index = index[valid]
        
        # ----- Compute transform

        # Compute point mean
        mdat = np.mean(dat_matched, 1)
        mref = np.mean(ref[:, index], 1)

        # Use SVD for transform computation
        C = np.transpose(dat_matched.T-mdat) @ (ref[:, index].T - mref)
        u, s, vh = np.linalg.svd(C)
        Ri = vh.T @ u.T
        Ti = mref - Ri @ mdat

        # Apply transformation to points
        dat_filt = Ri @ dat_filt
        dat_filt = np.transpose(dat_filt.T + Ti)

        # Update global transformation
        R = Ri @ R
        t = Ri @ t + Ti.reshape(2, 1)

        # Stop when no more progress
        if abs(olddist-meandist) < thres:
            break

        # store mean residual error to check progress
        olddist = meandist

    print("finished with mean point corresp. error {:f}".format(meandist))

    return R, t, meandist

