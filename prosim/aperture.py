"""
Aperture Module - aperture.py
=====

Provides :

    1. aperture constructor
    2. circular mask
    3. calculate    
"""
import numpy as np
from waves import Wave

class Aperture(): 
    """Class for aperture object.

        Methods:
        =====
        __init__ : constructor, Initializes aperture object with attribute(s).

        circular_mask : Returns circular cutout for array-like object (2D).

        calculate : Applies circular aperture to wave profile.

        """
    def __init__(self, wave: Wave):
        """Initialize aperture object.

        Attributes:
        =====
        wave (Wave) : target `Wave` object from `wave.py`.
        """
        self.__Nx = wave.Nx #samples to get center position

    def circular_mask(self, radius: int):
        """Return circular cutout for array-like object (2D).

        Attributes:
        =====
        radius (int, scalar) :  aperture radius, must be <= N/2
        """
        center = (int(self.__Nx/2), int(self.__Nx/2))
        Y, X = np.ogrid[:self.__Nx, :self.__Nx]
        distance = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
        mask = distance <= radius

        return mask
        
    def calculate(self, wave: Wave, radius: int):
        """Apply circular aperture to wave profile.

        Attributes:
        =====
        wave (Wave) : target `Wave` object from `wave.py`.
                                        
        radius (int, scalar) : aperture radius, must be <= `Wave.Nx/2`
        """

        wave.set_wp(wave.wp*self.circular_mask(radius)) #apply mask

        return 
