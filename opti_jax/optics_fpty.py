#!/usr/bin/env python
"""
Optics for simulating / solving Fourier Phytography illumination.

Hazen 2026.01
"""


import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax

import opti_jax.optics as optics


class OpticsFPty(optics.OpticsBF):
    """
    Fourier Phytography illumination.
    """
    def __init__(self, ni = 20, nj = 50, **kwds):
        """
        Default is 1000 iterations for solving (ni * nj).
        """
        super().__init__(**kwds)
        
        self.ni = ni
        self.nj = nj
        

    def plot_aperture(self, axs, axy, rscaler = 1.0):
        """
        Overlay aperture on FT of an image.
        """
        d0 = rscaler*2*self.kmax/self.dk0
        d1 = rscaler*2*self.kmax/self.dk1
        colors = ["red", "green"]
        for i, sign in enumerate([1,-1]):
            cxy = (sign*axy[1] + self.shape[1]//2, sign*axy[0] + self.shape[0]//2)
            circ = matplotlib.patches.Ellipse(cxy, d1, d0, edgecolor = colors[i], facecolor = "none", linestyle = "-")
            axs.add_patch(circ)


    def plot_ft_and_aperture(self, axs, img, axy, vmax = 4.0, rscaler = 1.0):
        """
        Plot FT of an image with the aperture overlaid.
        """
        axs.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1.0e-3), cmap = "gray", vmin = 0, vmax = vmax)
        self.plot_aperture(axs, axy, rscaler = rscaler)


    def plot_illumination_vectors(self, rxy, scale = 2.0, figsize = (5,5)):
        """
        Plot illumination vectors in k space.
        """
        fig, axs = plt.subplots(1, 1, figsize = (figsize[0], figsize[1]))

        for xy in rxy:
            axs.plot(self.dk0*xy[0], self.dk1*xy[1], "o", ms = 4)

        axs.plot([-scale*self.kmax, scale*self.kmax], [0, 0], ":", color = "gray")
        axs.plot([0, 0], [-scale*self.kmax, scale*self.kmax], ":", color = "gray")

        # Objective kmax.
        circ = plt.Circle((0.0, 0.0), radius = self.kmax, edgecolor="green", facecolor = "none", linestyle = ":")
        axs.add_patch(circ)
        
        plt.show()

        
    def solve_tv(self, Y, illm, lval = 1.0e-6, learningRate = 1.0e-1, order = 2, verbose = True):
        """
        Defaults tuned for fourier phytography.
        """
        return super().solve_tv(Y, illm, lval = lval, learningRate = learningRate, order = order, verbose = verbose)


    def y_pred(self, xrc, rxy):
        """
        Return images for each of the illumination 'angles'.

        rxy is the (integer) shift value to use in k space for each illumination angle.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for rx, ry in rxy:
            tmp.append(self.intensity(self.from_fourier(jnp.roll(xrcFT, (rx,ry), (0,1)) * self.mask)))
        return jnp.array(tmp)

    
class OpticsFPtyVP(optics.OpticsBFVP, OpticsFPty):
    """
    Fourier Phytography illumination with variable pupil.
    """
    def solve_tv(self, Y, illm, lval = 1.0e-6, lvalp = 1.0e-2, learningRate = 1.0e-1, order = 2, verbose = True):
        """
        Defaults tuned for fourier phytography.
        """
        x, nv = super().solve_tv(Y, illm, lval = jnp.array([lval, lvalp]), learningRate = learningRate, order = order, verbose = verbose)
        x = jnp.mod(x + jnp.pi, 2*jnp.pi) - jnp.pi
        return x, nv


    def x0(self, Y):
        """
        Initialize w/ additional term for pupil function.
        """
        yave = jnp.average(Y, axis = 0)
        return jnp.array([yave, jnp.zeros_like(yave), jnp.zeros_like(yave)])


    def y_pred(self, xrc, rxy):
        """
        Return images for each of the illumination 'angles'.

        rxy is the (integer) shift value to use in k space for each illumination angle.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        pupilFn = self.mask*jnp.exp(1j*xrc[2])
        for rx, ry in rxy:
            tmp.append(self.intensity(self.from_fourier(jnp.roll(xrcFT, (rx,ry), (0,1)) * pupilFn)))
        return jnp.array(tmp)
