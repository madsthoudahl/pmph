# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:37:10 2015

@author: Mads Thoudahl

"""
import numpy as np


materials = {
  # material number: [name, attenuation coefficient equation]
  0:["vacuum"  , 0 ],
  1:["hydrogen", 1 ],
  2:["helium"  , 2 ],
  3:["iron"    , lambda e: 1**-e ]  #how does this equation really look??
}


def randomscenegen( x0, y0, z0, # position lower left
                    x1, y1, z1, # position upper right
                    xs, ys, zs  # resolution
                  ):
    """ Generates an axis aligned scene, containing
    axis aligned boxes """
    matscene = np.random.randint(0, len(materials), xs*ys*zs)
    cornerpositions = [(x0, y0, z0),(x1, y1, z1)]

    return (cornerpositions, matscene)
