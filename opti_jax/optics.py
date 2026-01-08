#!/usr/bin/env python
"""
Optics base class.

Hazen 2026.01
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy


class Optics(object):
    """
    Base class for JAX optics.
    """
    def __init__(self, NA = None, pixelSize = None, shape = None, wavelength = None, **kwds):
        super().__init__(**kwds)

        self.NA = NA
        self.pixelSize = pixelSize
        self.shape = shape
        self.wavelength = wavelength

        self.kmax = self.NA/self.wavelength

        self.dk0 = 1.0/(self.shape[0] * self.pixelSize)
        self.dk1 = 1.0/(self.shape[1] * self.pixelSize)
        self.rmax0 = self.kmax/self.dk0
        self.rmax1 = self.kmax/self.dk1

        [self.g0,self.g1] = np.mgrid[-self.shape[0]//2 : self.shape[0]//2, -self.shape[1]//2 : self.shape[1]//2]

        # Real space vectors.
        #self.r0 = self.g0 * self.pixelSize
        #self.r1 = self.g1 * self.pixelSize

        # K space vectors.
        #self.k0 = self.dk0 * self.g0
        #self.k1 = self.dk1 * self.g1
        
        k0 = self.dk0 * self.g0
        k1 = self.dk1 * self.g1
        self.k = np.sqrt(k0 * k0 + k1 * k1)

        tmp = 1.0/self.wavelength
        self.kz = np.lib.scimath.sqrt(tmp * tmp - self.k * self.k)
        self.r = self.k/self.kmax
        self.kz[(self.r > 1.0)] = 0.0

        #self.npixels = np.sum(self.r <= 1)
        #self.norm = math.sqrt(self.r.size)

        # Mask for maximum pass frequency.
        self.mask = np.ones(self.shape)
        self.mask[(self.r > 1.0)] = 0.0

        # JAX array conversion.
        self.g0 = jnp.array(self.g0)
        self.g1 = jnp.array(self.g1)
        self.kz = jnp.array(self.kz)
        self.mask = jnp.array(self.mask)

        
    def from_fourier(self, imageFt):
        return jnp.fft.ifft2(jnp.fft.ifftshift(imageFt))

    
    def intensity(self, image):
        return jnp.abs(image * jnp.conj(image))

    
    def kvalue_range(self):
        return [self.kmax/self.dk0, self.kmax/self.dk1]

    
    def to_fourier(self, image):
        return jnp.fft.fftshift(jnp.fft.fft2(image))


class OpticsBF(Optics):
    """
    Brightfield illumination.
    """
    def check_pattern(self, pat):
        """
        (Visual) check that illumination pattern is within the objective aperture.
        """
        xrc = [jnp.ones(self.shape), jnp.zeros(self.shape)]
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        sumFT = jnp.zeros_like(xrcFT)
        for rx, ry in pat:
            sumFT = sumFT + jnp.roll(xrcFT, (rx,ry), (0,1))
        return sumFT


    def illuminate(self, xrc, k0, k1):
        """
        xrc is [real, complex] as not all Optax solvers handle complex numbers.
        """
        tmp = xrc[0] + 1j*xrc[1]
        return tmp * jnp.exp(1j * (k0 * self.g0 + k1 * self.g1))
    

    def make_bf_pattern(self, maxNA, ik0, ik1):
        """
        ik0 and ik1 are expected to be integer arrays.
        """
        pat = []
        for k0 in ik0:
            for k1 in ik1:
                kk = np.sqrt(k0*k0*self.dk0*self.dk0 + k1*k1*self.dk1*self.dk1)
                if (kk > maxNA/self.wavelength):
                    continue
                pat.append([int(k0), int(k1)])
                
        return jnp.array(pat)
        
        
    def patterned_illumination(self, xrc, rxy):
        """
        xrc is [real, complex] as not all Optax solvers handle complex numbers.
        For efficiency we roll the fourier transform instead of recalculating the illumination for all k values.
        """
        pim = jnp.zeros(xrc[0].shape, dtype = float)
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for rx, ry in rxy:
            pim = pim + self.intensity(self.from_fourier(jnp.roll(xrcFT, (rx,ry), (0,1))*self.mask))
        return pim/float(len(rxy))


    def plot_pattern(self, pat, figsize = (5,5)):
        """
        Plot illumination pattern in k space.
        """
        fig = plt.figure(figsize = figsize)
        plt.plot(self.dk0*pat[:,0], self.dk1*pat[:,1], "o", ms = 4)
        plt.plot([-self.kmax, self.kmax], [0, 0], ":", color = "gray")
        plt.plot([0, 0], [-self.kmax, self.kmax], ":", color = "gray")
        plt.show()
