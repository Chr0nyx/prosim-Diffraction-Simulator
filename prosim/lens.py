"""
Lens Module - lens.py
=====

Provides :

    1. lens constructor
    2. phase 
    3. calculate 

"""
import numpy as np
from waves import Wave
from functions import *


class Lens():
    """Class for lens object.

        Methods:
        =====
        __init__ : constructor, Initializes lens object with attribute(s).

        phase : Returns phase term depending on wave object.

        calculate : Applies phase to wave object.

        """

    def __init__(self, focal_length: float):
        """Initialize lens object.

        Attributes:
        =====
        focal_length (float, scalar) : Focal length of lens object.
                                        Represents the distanze from the lens to 
                                        position of maxium beam focus.
        """
        self.__fc = focal_length

    @property
    def focal_length(self):
        return self.__fc
        
    def phase(self, wave: Wave, λ: float):
        """Return phase.

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from `wave.py`
                      Describes the entire wave at origin.   
        λ (float, scalar) : Wave length
                             Spatial perodicity of wave.                
        """
        X, Y = wave.coordinates
        return 1j*np.pi*(X**2+Y**2)/(self.__fc*λ)

    def calculate_continouos(self, wave: Wave, λ: float):
        """Return electric field with applied phase shift.

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from `wave.py`
                      Describes the entire wave at origin.  
        λ (float, scalar) : Wave length        
        """
        phi = self.phase(wave, λ)        
        return wave.wp*np.exp(-phi)


    def calculate_pulse(self, wave: Wave):
        """Return pulsed electric field with applied phase shift.

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from `wave.py`
                      Describes the entire wave at origin.  
        """

        pλ = split(wave.λ)
        m = 1.0

        for λ in pλ:
            phi = self.phase(wave, λ)
            m = m*np.exp(-phi)

        return wave.wp*m
