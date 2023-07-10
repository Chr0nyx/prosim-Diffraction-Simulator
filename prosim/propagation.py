"""
Propagation Module - propagation.py
=====

Provides :

    1. propagation constructor
    2. prop_TF
    3. prop_IR
    4. propagate_2D
    5. calculate_continous
    6. calculate_pulse
    
"""
import numpy as np
from numpy.fft import ifft2, fft2, ifftshift, fftshift
from functions import progress_bar, stripe, split
from waves import Wave

class Propagation():
    """Class for free space propagation object.

        Methods:
        =====
        __init__ : constructor, Initializes propagation object with attribute(s). 

        prop_TF : Returns transfer function matrix for propagate_2D-function.

        prop_IR : Returns impulse matrix for propagate_2D-function.

        propagate_2D : Performs propagation of wave.

        calculate_continous : Performs propagation of wave with respect to 
                              propagation axis for monochromatic wave.

        calculate_pulse : Performs propagation of wave with respect to 
                          propagation axis for polychromatic wave.
        """

    def __init__(self, Z: float, Nz: int):
        """Initialize propagation object.

        Attributes:
        =====
        Z (float, scalar) : Propagation distance
                             Position the wave should propgate to. 

        Nz (int, scalar) : Propagation resolution
                           Amount of propagation distance samples. 
                           Size increases accuracy and computation time.
        """
        self.__Z = Z
        self.__Nz = Nz
        self.__z = np.linspace(0, self.__Z, self.__Nz)
    
    #Transfer Function Approach
    def prop_TF(self, wave: Wave, λ: float, k: float, z: float): 
        """Return transfer function (TF) for free space propagation.

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from `wave.py`
                      Describes the entire wave at origin.                                     

        λ (float, scalar) : Wave length
                             Spatial perodicity of wave.

        k (float, scalar) : Wave number
                             Scaled spatial perodicity of wave.
        
        z (float, scalar) : Propagation distance at which propagation should be
                             computed.
                             
        """
        
        fx = np.linspace(-1/(2*wave.dx), 1/(2*wave.dx)-1/wave.L, wave.Nx)
        FX, FY = np.meshgrid(fx, fx)

        #Make transfer function matrix
        H = np.exp(-1j*np.pi*λ*z*(FX**2+FY**2)) * np.exp(1j*k*z)
            
        return fftshift(H)   

    #Impulse Response Approach
    def prop_IR(self, wave: Wave, λ: float, k: float, z: float):
        """Return impulse response (IR) for free space propagation.

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from `wave.py`
                      Describes the entire wave at origin.                                    

        λ (float, scalar) : Wave length
                             Spatial perodicity of wave.

        k (float, scalar) : Wave number
                             Scaled spatial perodicity of wave.
        
        z (float, scalar) : Propagation distance at which propagation should be
                             computed.
        """
    
        X, Y = wave.coordinates

        #Make impulse response matrix
        h = np.exp(1j*k*(X**2+Y**2)/(2*z))/(1j*λ*z) * np.exp(1j*k*z)
        
        return fft2(fftshift(h))*wave.dx**2

    def propagate_2D(self, wave: Wave, λ: float, k: float, z: float):
        """Return propagation for 2D (array-like) wave profile.

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from `wave.py`
                      Describes the entire wave at origin.                                  

        λ (float, scalar) : Wave length
                             Spatial perodicity of wave.

        k (float, scalar) : Wave number
                             Scaled spatial perodicity of wave.
        
        z (float, scalar) : Propagation distance at which propagation should be
                             computed.
        """
        #Make 2D Fourier transform of wave profile
        U1 = fft2(fftshift(wave.wp))
        U2 = np.zeros((wave.Nx, wave.Nx), dtype=complex)
        #Decide according sampling criteria
        if(wave.dx > λ*z/wave.L):
            U2 = ifftshift(ifft2(U1*self.prop_TF(wave, λ, k, z)))
        else:
            U2 = ifftshift(ifft2(U1*self.prop_IR(wave, λ, k, z)))

        
        return U2

    def calculate_continuous(self, wave: Wave, λ: float, k: float):
        """Return propagated beam as (2D) array-like for monochromatic wave. 

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from wave.py
                      Describes the entire wave at origin.                                    

        λ (float, scalar) : Wave length
                             Spatial perodicity of wave.

        k (float, scalar) : Wave number
                             Scaled spatial perodicity of wave.
        """

        Erz = np.zeros((int(wave.Nx/2), self.__Nz), dtype=complex)

        for i, z_i in enumerate(self.__z):  
        
            progress_bar(i, self.__Nz, 20)

            if z_i == 0:
                Erz[:, i] = stripe(wave.wp)
            else:
                Erz[:, i] = stripe(self.propagate_2D(wave, λ, k, z_i))

        return Erz
    
    
    def calculate_pulse(self, wave: Wave):
        """Return propagated beam as (2D) array-like for polychromatic wave. 
           (corresponds to a pulse).

        Attributes:
        =====
        wave (Wave) : wave object as represented by `Wave` class from `wave.py`
                      Describes the entire initial wave. 
        """

        E0 = wave.wp #Initial wave profile
        Ez = [] #Prepare empty container

        l = int(wave.Nx/2) #Get wave profile length
        t0=self.__Z/wave.v_ph 
        t = np.linspace(t0-wave.T/2, t0+wave.T/2, self.__Nz) #Make array for propagation range

        #take only frequencies > 0
        pω = split(wave.ω)
        psp = split(wave.sp)
        pλ = split(wave.λ)
        pk = split(wave.k)
                
        for i, t_i in enumerate(t):
            
            progress_bar(i, len(t), title='pulse') #Start progress bar
            propagated = np.zeros(l, dtype=complex) #Initialize complex zero container

            for j, s_j in enumerate(psp):

                wave.set_wp(E0*np.abs(s_j)) #Account for spectral weight
                Ezj = self.propagate_2D(wave, pλ[j], pk[j], self.__Z) #Evaluate actual propagation
                wave.set_wp(Ezj * np.exp(1j * (pω[j] * t_i - np.angle(s_j))))
                propagated += stripe(wave.wp)

            Ez.append(propagated)

        progress_bar(len(t), len(t))
        #notify_user('Prosim:', 'Compute propagation: completed')

        
        return np.transpose(np.asanyarray(Ez))

    @property
    def Z(self):
        return self.__Z
