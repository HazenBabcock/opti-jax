#!/usr/bin/env python
"""
Optics for simulating / solving from a (bright-field) Z stack.

Hazen 2026.01
"""


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import opti_jax.optics as optics


class OpticsZStack(optics.OpticsBF):
    """
    Z (focus) stack.
    """
    def __init__(self, ni = 20, nj = 50, **kwds):
        """
        Default is 1000 iterations for solving (ni * nj).
        """
        super().__init__(**kwds)

        self.ni = ni
        self.nj = nj


    def compute_loss_illumination(self, x, Y, sData, illm):
        """
        Illumination loss function.
        """
        yPred = self.y_pred(x, [sData[0], illm, sData[1]])
        loss = jnp.mean(optax.l2_loss(yPred, Y))
        return loss
    

    def make_bf_pattern(self, maxNA, ik0, ik1):
        """
        Approximate uniform illumination over an aperture with multiple point sources.
        
        ik0 and ik1 are expected to be integer arrays.
        """
        rxy = super().make_bf_pattern(maxNA, ik0, ik1)
        return [rxy, jnp.ones(len(rxy))]


    def plot_pattern(self, pat, intensity, mscale = 2000, figsize = (5,5)):
        """
        Plot pattern with intensity values indicated by circle size.
        """

        fig, axs = plt.subplots(1, 1, figsize = (5,5))
        axs.scatter(self.dk0*pat[:,0], self.dk1*pat[:,1], s = mscale*intensity, facecolor = "none", edgecolor = "black")
        axs.plot([-1.1*self.kmax, 1.1*self.kmax], [0, 0], ":", color = "gray")
        axs.plot([0, 0], [-1.1*self.kmax, 1.1*self.kmax], ":", color = "gray")
    
        # Objective kmax.
        circ = plt.Circle((0.0, 0.0), radius = self.kmax, edgecolor="green", facecolor = "none", linestyle = ":")
        axs.add_patch(circ)
        
        plt.show()
    
    
    def solve_tv(self, Y, sData, lval = 1.0e-5, learningRate = 1.0e-1, order = 2, verbose = True, x0 = None):
        """
        Defaults tuned for a bright field focus stack.
        """
        return super().solve_tv(Y, sData, lval = lval, learningRate = learningRate, order = order, verbose = verbose, x0 = x0)


    def x0(self, Y):
        """
        Assume middle image has the best focus.
        """
        ybest = Y[len(Y)//2]
        return jnp.array([ybest, jnp.zeros_like(ybest)])


    def y_pred(self, xrc, sData):
        """
        Return images for each of the illumination z values.
        """
        [rxy, intensities, zvals] = sData
        
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for zv in zvals:
            zshift = jnp.exp(1j * 2.0 * jnp.pi * self.kz * zv)
            pim = jnp.zeros(self.shape)
            for i in range(len(rxy)):
                pim += self.intensity(self.from_fourier(jnp.roll(xrcFT, rxy[i], (0,1)) * zshift * self.mask)) * intensities[i]
            tmp.append(pim/np.sum(intensities))
        return jnp.array(tmp)

    
    def y_pred_ft(self, xrc, sData):
        """
        Return images for each of the illumination z values, fourier space.
        """
        [rxy, intensities, zvals] = sData
        
        tmp = []
        xrcFT = xrc[0] + 1j*xrc[1]
        for zv in zvals:
            zshift = jnp.exp(1j * 2.0 * jnp.pi * self.kz * zv) * self.mask
            pim = jnp.zeros(self.shape)
            for i in range(len(rxy)):
                pim += self.intensity(self.from_fourier(jnp.roll(xrcFT, rxy[i], (0,1)) * zshift)) * intensities[i]
            tmp.append(pim/np.sum(intensities))
        return jnp.array(tmp)


class OpticsZStackVP(OpticsZStack, optics.OpticsBFVP):
    """
    Z (focus) stack with variable pupil.
    """
    def solve_tv(self, Y, sData, lval = 1.0e-5, lvalp = 1.0e-2, learningRate = 1.0e-1, order = 2, verbose = True, x0 = None):
        """
        Defaults tuned for a bright field focus stack with variable pupil function.
        """
        x, nv = super().solve_tv(Y, sData, lval = jnp.array([lval, lvalp]), learningRate = learningRate, order = order, verbose = verbose, x0 = x0)
        x2 = jnp.mod(x[2] + jnp.pi, 2*jnp.pi) - jnp.pi
        return jnp.array([x[0], x[1], x2]), nv


    def x0(self, Y):
        """
        Initialize w/ additional term for pupil function.
        """
        ybest = Y[len(Y)//2]
        return jnp.array([ybest, jnp.zeros_like(ybest), jnp.zeros_like(ybest)])
    

    def y_pred(self, xrc, sData):
        """
        Return images for each of the illumination z values.
        """
        [rxy, intensities, zvals] = sData

        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        pupilFn = self.mask*jnp.exp(1j*xrc[2])
        for zv in zvals:
            zshift = jnp.exp(1j * 2.0 * jnp.pi * self.kz * zv) * pupilFn
            pim = jnp.zeros(self.shape)
            for i in range(len(rxy)):
                pim += self.intensity(self.from_fourier(jnp.roll(xrcFT, rxy[i], (0,1)) * zshift)) * intensities[i]
            tmp.append(pim/np.sum(intensities))
        return jnp.array(tmp)
