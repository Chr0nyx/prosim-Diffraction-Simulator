"""
Wave Module - wave.py
=====

Provides :

    1. wave object constructor
    2. set_wp
    3. set_sp
    4. set_Ɛr
    5. set_µr
    6. set_image
    7. properties

"""
import numpy as np
import scipy.constants as spc
from PIL import Image, ImageOps

class Wave():
    """Class for wave object.

        Methods:
        =====
        __init__ : constructor, Initializes wave object with attribute(s). 

        set_wp : Set wave profile.
        
        set_sp : Set spectrum.
        
        set_µ_r : Set permeability.
        
        set_Ɛ_r : Set permittivity.

        set_image : Set wave profile to img file.
        
        properties : Access object properties.
        """

    def __init__(self, L: float, Nx: int, T: float, Nt: int, µ_r=1.0, Ɛ_r=1.0):
        """Initialize wave object.

        Attributes:
        =====
        L (float, scalar) : Grid length
                             Size of grid for electrical field of wave.
                             Limits set as: -L/2 to +L/2

        Nx (int, scalar) : Grid Resolution
                           Amount of spatial samples to discretize electrical
                           field.
                           Size increases accuracy and computation time.

        T (float, scalar) : Pulse Time. Range of observed time. 

        Nt (int, scalar) : Time Resolution 
                           Amount of temporal samples to discretize pulse time.
                           Size increases accuracy and computation time.

        µ_r (float, scalar) : Permeability, default = 1.0, int
                       
        Ɛ_r (float, scalar) : Permittivity, default = 1.0, int
        """
        #Spatial Resolution
        self.__L = L 
        self.__Nx = int(Nx) 
        self.__dx = self.__L/self.__Nx #Space Step
        self.__x = np.linspace(-L/2, L/2+self.__dx, self.__Nx) #Carthesian Coordinate Base
        self.__X, self.__Y = np.meshgrid(self.__x,self.__x) #XY-Grid
        self.__extent = [self.__x[0], self.__x[-1]+self.__dx, self.__x[0], self.__x[-1]+self.__dx]
        self.__wp = np.zeros((self.__Nx,self.__Nx)) #Radially Symmetric Field Distribution

        #Temporal Resolution
        self.__T = T
        self.__Nt = int(Nt)
        self.__ω = np.zeros(self.__Nt) #Frequency Coordinates
        self.__sp = np.zeros(self.__Nt) #Pulse Spectrum

        #Medium
        self.__µ_r = µ_r #lin. iso. hom. Permeability 
        self.__Ɛ_r = Ɛ_r #lin. iso. hom. Permittivity 

        #Wave properties
        self.__v_ph = spc.speed_of_light/np.sqrt(self.__µ_r*self.__Ɛ_r) #Phase Velocity
        self.__λ = np.zeros(self.__Nt) #Wave Length
        self.__k = np.zeros(self.__Nt) #Wave Number

    def set_wp(self, wp: complex):
        """Set wave profile. 

        Arguments:
        =====
        wp (complex, array-like) : Wave profile.
                                   Represents wave's electric field distribution.
        """
        self.__wp = wp

    def set_sp(self, sp: complex, ω: float):
        """Set spectrum. 

        Arguments:
        =====
        sp (complex, array-like) : Frequency spectrum
                                   Represents frequency spectrum of wave, can be
                                   scalar or array-like (1D).
        
        ω (float, array-like) : Angular frequency ω=2*Pi*f 
                                 Represents the wave's frequency with respect
                                 to spectrum sp, can be scalar of array-like (1D).
        """
        self.__sp = sp
        self.__ω = ω
        self.__λ = 2*np.pi*self.__v_ph/self.__ω
        self.__k = 2*np.pi/self.__λ 

    def set_Ɛr(self, Ɛ_r: float):
        """Set permittivity.

        Arguments:
        =====
        Ɛ_r (float, scalar) : Relative permettivity
                               Represents the polarisation of the material the
                               wave propagates through.
        """
        self.__Ɛ_r = Ɛ_r
        self.__v_ph = spc.speed_of_light/np.sqrt(self.__µ_r*self.__Ɛ_r) 

    def set_µr(self, µ_r: float):
        """Set permeability.

        Arguments:
        =====
        µ_r (float, scalar): Relative permeability
                              Represents the magnetisation of the material the
                              wave propagates through.
        """
        self.__µ_r = µ_r
        self.__v_ph = spc.speed_of_light/np.sqrt(self.__µ_r*self.__Ɛ_r) 

    def set_image(self, file: str):
        """Set wave profile to image 

        Arguments:
        =====
        file (string): Absolute path to image file.
        """
        rgb = Image.open(file)
        gray = ImageOps.grayscale(rgb)
        img_array = np.array(gray)/255
        N_img = len(img_array)

        self.__Nx = int(N_img) 
        self.__dx = self.__L/self.__Nx #Space Step
        self.__x = np.linspace(-self.__L/2, self.__L/2+self.__dx, self.__Nx) #Carthesian Coordinate Base
        self.__X, self.__Y = np.meshgrid(self.__x,self.__x) #XY-Grid
        self.__extent = [self.__x[0], self.__x[-1]+self.__dx, self.__x[0], self.__x[-1]+self.__dx]
        self.__wp = img_array 
        
        return

    @property
    def extent(self):
        return self.__extent
    @property
    def irr(self):
        return self.__wp.real**2 + self.__wp.imag**2
    @property
    def ω(self):
        return self.__ω
    @property
    def sp(self):
        return self.__sp
    @property
    def coordinates(self):
        return self.__X, self.__Y 
    @property
    def wp(self):
        return self.__wp
    @property
    def dx(self):
        return self.__dx
    @property
    def L(self):
        return self.__L
    @property 
    def Nx(self):
        return self.__Nx
    @property
    def T(self):
        return self.__T
    @property 
    def Nt(self):
        return self.__Nt
    @property
    def λ(self):
        return self.__λ
    @property
    def k(self):
        return self.__k
    @property
    def v_ph(self):
        return self.__v_ph

    