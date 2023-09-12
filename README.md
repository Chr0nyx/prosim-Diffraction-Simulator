# PROSIM - Python Propagation (and Diffraction) Simulator
---
Prosim is a tool for numerical simulation of electromagnetic waves using the angular spectrum method.
---

Requirements in respective file:
+ anaconda-client 1.11.0
+ anaconda-navigator 2.3.1
+ numpy 1.21.5
+ scipy 1.9.1
+ plyer 2.1.0
+ Pillow 9.2.0

Package-Structure:

![package](https://github.com/hakimtayari/prosim-Diffraction-Simulator/assets/88373056/f716284b-b4ae-4d02-a3cf-41650b261139)

## A chunk of Math

The Maxwell equations for the electromagnetic field provide the basis for scalar diffraction. They enable the derivation of the scalar Helmholtz equation.

$$ \Delta E(r) + k^2  E(r) = 0 $$      
     
To which the Fresnel-Kirchhoff integral (below) provides a valid solution in the context of the phenomenon of scalar diffraction.
 
$$  E(x, y, z) = -\dfrac{i}{\lambda}  \iint\limits_{aperture} E(x\', y\', 0) $$ $$ \dfrac{e^{ikR}}{R}   \dfrac{1+cos(\gamma)}{2} dx\' dy\' $$
    
The Fresnel-Kirchhoff integral can be evaluated more efficiently by transforming it into the Fresnel-approximation.
It describes the diffraction (and therefore propagation after the interacting aperture) of a monochromatic em wave at a propagation distance z with respect to small angles between the propagational axis and the beam radius R.

$$ E(x,y,z) \cong  -i\Gamma \iint\limits_{aperture} E(x\', y\', 0) $$ $$ \; e^{i\dfrac{k}{2z}(x\'^2+y\'^2)} \; e^{-i\dfrac{k}{z}(xx\'+y y\')}dx\'dy\ $$

$$ with \; \Gamma = \dfrac{e^{ikz}e^{i\dfrac{k}{2z}(x^2+y^2)}}{\lambda z} $$

Using an evenly sampled symmetric grid a **two dimensional Fourier transform** of the Fresnel approximation can be used to numerically evaluate the diffraction process and calculate the resulting electrical field at the specified distance. 

## Usage

Example: reproduction of [Gaussian Beam](https://en.wikipedia.org/wiki/Gaussian_beam)

```python

#'Main' file for computation

import matplotlib as mpl
# Use the pgf backend (must be set before pyplot imported)
#mpl.use('pgf')
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as cs
from numpy.fft import ifft2, fft2, ifftshift, fftshift
import prosim as ps
from prosim import functions as fun
from prosim import parameters as par

w1 = ps.Wave(par.X, par.Nx, par.T, par.Nt)
p1 = ps.Propagation(par.Z, par.Nz)

#analytic beam waist:
z_r = np.pi*par.w_0**2/par.λ0
w_of_z = par.w_0 * np.sqrt(1+par.z**2/z_r**2)
#numeric wais:
w_of_z_num = np.zeros(par.Nz)

#make wave profile
X, Y = w1.coordinates
custom_wave = np.select([(np.exp(-(X**2+Y**2)/par.w_0**2) <= 1), True], [np.exp(-(X**2+Y**2)/par.w_0**2),0])
w1.set_wp(custom_wave)


#Calcs
Erz = p1.calculate_continuous(w1, par.λ0, par.k0)
Irz_norm = fun.normalize_beam(Erz)
w_of_z_num = fun.evaluate_beam_radius(Erz, w1.dx)


fig, ax = plt.subplots(3)
fig.tight_layout(pad=1.5)
figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()


extent = [0, par.Z*1e3, 0, par.X*1e3]
#uncomment to see curvature
p0=ax[0].imshow(Erz.real**2, cmap='jet', extent=extent, aspect='auto')

#uncomment to see normalized beam
p1=ax[1].imshow(Irz_norm, cmap='jet', extent=extent, aspect='auto')

#uncomment to see beam waist
ax[2].plot(par.z*1e3, w_of_z_num*1e3, 'b+', par.z*1e3, w_of_z*1e3, 'r-', linewidth='0.7')
ax[2].grid(which='major', axis='both')
ax[2].legend(['numeric','analytic'])
ax[2].set_xlabel('Propagation distance z in mm')
ax[2].set_ylabel('Beam Radius w in mm')
ax[2].set_title('Beam radius as a function of propagation distance for Gaussian beam \
                \n \n $N_x$ =%.0f, $N_z$ =%.0f, $w_0$=%.0fmm, $\lambda_0$=%.0fmm' % (par.Nx, par.Nz, par.w_0/par.milli, par.λ0/par.milli))


#uncomment to see colorbar next to imshow
fig.colorbar(p0, ax=ax[0])
fig.colorbar(p1, ax=ax[1])


plt.show()
```

with 

```python
"""
Parameters Module - parameters.py
=====

Provides :

    Parameters for simulation
"""
import numpy as np
from scipy import constants as cs
from prosim.prefix import *
c0 = cs.speed_of_light

milli= 10e-3

#-------RESOLUTION--------#
#Spatial Resolution
Nx = 400
#Grid Size
X = 400*milli

#Propagation Resolution
Nz = 400
#Propagation Distance
Z = 400*milli
z = np.linspace(0, Z, Nz)
dz = Z/Nz

#----- MONOCHROMATIC GAUSSIAN ------#
#Initial Beam Waist (if Gaussian)
#wiki: w_0 = 40*milli
w_0 = 40*milli
#Wavelength 
λ0 = 30*milli
#Monochromatic Frequency 
f0 = c0/λ0
ω0 = 2*np.pi*f0
#Wavenumber
k0 = 2*np.pi/λ0

```

Results in:

![gaus_beam](https://github.com/hakimtayari/prosim-Diffraction-Simulator/assets/88373056/50f9a94f-0c83-4ca7-afc7-95b2f0ccd328)



## Shortcomings

The implementation is based on the excellent ***Computational Fourier Optics: A MATLAB Tutorial*** by **D. G. Voelz** and
***Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria*** also by **D. G. Voelz','M. C. Roggemann** which already delve quite deep into the drawbacks a numerical computation has using the angular spectrum method.
For the computation to *most* accurately model the diffraction (or propagation) the chosen samples and the parameterized wave in compliance with the [Fresnel-Approximation](https://en.wikipedia.org/wiki/Fresnel_diffraction) are of highest importance. 

## Improvements 


Feel free to fork, criticize or improve this project.
