#!/usr/bin/env python
"""
Optics for simulating / solving DPC illumination.

Hazen 2026.01
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import opti_jax.optics as optics


class OpticsDPC(optics.OpticsBF):
    """
    DPC illumination, nominally 4 patterns.
    """
    def __init__(self, ni = 20, nj = 50, **kwds):
        """
        Default is 1000 iterations for solving (ni * nj).
        """
        super().__init__(**kwds)
        
        self.ni = ni
        self.nj = nj
        
        
    def compute_loss_tv_order1(self, x, Y, pats, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.dpc_illumination(x, pats)
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order1(x)*lval
        return loss


    def compute_loss_tv_order2(self, x, Y, pats, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.dpc_illumination(x, pats)
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order2(x)*lval
        return loss

    
    def dpc_illumination(self, xrc, pats):
        """
        Return images for each of the DPC illumination patterns.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for pat in pats:
            tmp.append(self.patterned_illumination_ft(xrcFT, pat))
        return jnp.array(tmp)
    
        
    def make_dpc_patterns(self, maxNA, ik0, ik1):
        """
        The illumination pattern order should be checked to verify that it matches reality.

        This particular pattern corresponds to [270, 90, 0, 180] in the coordinate system
        of the 'Waller-Lab/DPC_withAberrationCorrection' project.
        """
        p0 = []
        p1 = []
        p2 = []
        p3 = []
        for k0 in ik0:
            for k1 in ik1:
                kk = np.sqrt(k0*k0*self.dk0*self.dk0 + k1*k1*self.dk1*self.dk1)
                if (kk > maxNA/self.wavelength):
                    continue

                if (k1 < -0.01):
                    p0.append([k0, k1])
                elif (k1 > 0.01):
                    p1.append([k0, k1])

                if (k0 < -0.01):
                    p2.append([k0, k1])
                elif (k0 > 0.01):
                    p3.append([k0, k1])
                    
        return jnp.array([jnp.array(p0), jnp.array(p1), jnp.array(p2), jnp.array(p3)])


    def plot_fit_images(self, Y, x, pats, vrange = 1.0e-2):
        """
        Plot DPC images and corresponding fit images.
        """
        YPred = self.dpc_illumination(x, pats)

        fig, axs = plt.subplots(3, len(Y), figsize = (4*len(Y), 12))
        for i in range(len(Y)):
            axs[0,i].imshow(Y[i], cmap = "gray", vmin = 0.0, vmax = 1.0)
            axs[1,i].imshow(YPred[i], cmap = "gray", vmin = 0.0, vmax = 1.0)
            axs[2,i].imshow(Y[i]-YPred[i], cmap = "gray", vmin = -vrange, vmax = vrange)
            
            for j in range(2):
                axs[j,i].set_xticks([])
                axs[j,i].set_yticks([])

        plt.show()
        

    def plot_patterns(self, pats, figsize = (5.5,5)):
        """
        Plot illumination patterns in k space.
        """
        fig, axs = plt.subplots(1, len(pats), figsize = (figsize[0]*len(pats), figsize[1]))

        for ii, pat in enumerate(pats):
            axs[ii].plot(self.dk0*pat[:,0], self.dk1*pat[:,1], "o", ms = 4)
            axs[ii].plot([-1.1*self.kmax, 1.1*self.kmax], [0, 0], ":", color = "gray")
            axs[ii].plot([0, 0], [-1.1*self.kmax, 1.1*self.kmax], ":", color = "gray")

            # Objective kmax
            circ = plt.Circle((0.0, 0.0), radius = self.kmax, edgecolor="green", facecolor = "none", linestyle = ":")
            axs[ii].add_patch(circ)
        
        plt.show()


    def tv_solve(self, Y, illm, lval = 1.0e-3, learningRate = 1.0e-2, order = 2, verbose = True):
        """
        Defaults tuned for DPC.
        """
        return super().tv_solve(Y, illm, lval = lval, learningRate = learningRate, order = order, verbose = verbose)
