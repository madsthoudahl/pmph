#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 31 14:14:15 2015

@author: Mads Thoudahl

"""
from xraysimphysics import randomscenegen, materials
from xraysimgeometry import *

scene_size = 10 #is cubed!
power = 3
energylevel = 4

asource = (-1,0,0, # position
          10,    # Power [Joules]
          80     # Energy level of xrays [keV]
         )

ascenedefs = (-0.5,-0.5,-0.5, # position lower left
          0.5, 0.5, 0.5, # position upper right
         scene_size,    # resolution1 (#pixels)
         scene_size,    # resolution2 (#pixels)
         scene_size,    # resolution3 (#pixels)
         # voxelcount = res1 * res2 * res3
        )

adetector = (1,-1,-1, # position lower left
            1, 1, 1, # position upper right
            10,    # resolution1 (#pixels)
            10,    # resolution2 (#pixels)
           )

def xraysim_benchmark(
      sourcelist = [asource],
      scenedefs = ascenedefs,
      detectorlist = [adetector]
      ):
    print "xraysim execution"

    # generate a random scene
    scenecorners, scenematerials = randomscenegen(scenedefs)

    for source in sourcelist:
        # preprocess the scene physics
        # building a map of attenuation coefficients
        sceneattenuates =  np.zeros(scenematerials.shape)
        for material in materials:
            sceneattenuates += (scenematerials == material) * materials[material](source[energylevel])

        #
        for detector in detectorlist:
            dx = detector[3] - detector[0]
            dy = detector[4] - detector[1]
            dz = detector[5] - detector[2]
            detector_norm =


    return 0


if __name__ == '__main__':
    xraysim_benchmark()