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


class OpticsDPCBase(optics.OpticsBF):
    """
    Base class for DPC illumination.
    """
    def __init__(self, ni = 20, nj = 50, **kwds):
        """
        Default is 1000 iterations for solving (ni * nj).
        """
        super().__init__(**kwds)

        self.ni = ni
        self.nj = nj


    def dpc_illumination_patterns(self, xrc, pats):
        """
        Return images for each of the DPC illumination patterns.

        Note that this is an approximation. We are adding the images in Fourier
        space and then doing in the inverse transform. This makes the different
        illuminations add coherently which is not correct, but seems to be a
        'good enough' approximation.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for rxy in pats:
            tmp.append(self.patterned_illumination_ft(xrcFT, rxy))
        return jnp.array(tmp)


    def dpc_illumination_patterns_no_roll(self, xrc, pats):
        """
        Return images for each of the DPC illumination patterns.

        This is "Truth" to test against.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for rxy in pats:
            tmp.append(self.patterned_illumination_no_roll_ft(xrcFT, rxy))
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
                    p0.append([int(k0), int(k1)])
                elif (k1 > 0.01):
                    p1.append([int(k0), int(k1)])

                if (k0 < -0.01):
                    p2.append([int(k0), int(k1)])
                elif (k0 > 0.01):
                    p3.append([int(k0), int(k1)])
                    
        return jnp.array([jnp.array(p0).astype(int), jnp.array(p1).astype(int), jnp.array(p2).astype(int), jnp.array(p3).astype(int)]).astype(int)


    def make_mask(self, rxy):
        """
        Make a Fourier space mask for an illumination pattern.
        """
        mask = np.zeros(self.shape)
        for rx, ry in rxy:
            mask += np.roll(self.mask, (-rx,-ry), (0,1))
        return jnp.array(mask/float(len(rxy)))

    
    def make_masks(self, pats):
        """
        Make a Fourier space mask for each of the illumination patterns.
        """
        masks = []
        for rxy in pats:
            masks.append(self.make_mask(rxy))
        return jnp.array(masks)

    
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


class OpticsDPC(OpticsDPCBase):
    """
    DPC illumination, nominally 4 patterns.
    """
    def plot_fit_images(self, Y, x, pats, vrange = 1.0e-2):
        """
        Plot DPC images and corresponding fit images.
        """
        super().plot_fit_images(Y, x, self.make_masks(pats), vrange = vrange)

    
    def solve_tv(self, Y, illm, lval = 1.0e-3, learningRate = 1.0e-2, optimizer = None, order = 2, verbose = True, x0 = None):
        """
        Defaults tuned for DPC.
        """
        masks = self.make_masks(illm)
        return super().solve_tv(Y, masks,
                                lval = lval,
                                learningRate = learningRate,
                                optimizer = optimizer,
                                order = order,
                                verbose = verbose,
                                x0 = x0)


    def y_pred(self, xrc, sData):
        """
        Return images for each of the DPC illumination pattern masks.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for mask in sData:
            tmp.append(self.intensity(self.from_fourier(xrcFT*mask)))
        return jnp.array(tmp)    
    

class OpticsDPCVp(optics.OpticsBFVp, OpticsDPCBase):
    """
    DPC illumination, nominally 4 patterns, with variable pupil function.

    Note that this is significantly slower as now we are no longer making
    the assumption that the coherent and incoherent images are the same.
    """
    def solve_tv(self, Y, illm, lval = 1.0e-3, lvalp = 1.0e-2, learningRate = 1.0e-1, optimizer = None, order = 2, verbose = True, x0 = None):
        """
        Defaults tuned for DPC with variable pupil function.
        """
        x, nv = super().solve_tv(Y, illm,
                                 lval = jnp.array([lval, lvalp]),
                                 learningRate = learningRate,
                                 optimizer = optimizer,
                                 order = order,
                                 verbose = verbose,
                                 x0 = x0)

        x2 = jnp.mod(x[2] + jnp.pi, 2*jnp.pi) - jnp.pi
        return jnp.array([x[0], x[1], x2]), nv


    def x0(self, Y):
        """
        Initialize w/ additional term for pupil function.
        """
        yave = jnp.average(Y, axis = 0)
        return jnp.array([yave, jnp.zeros_like(yave), jnp.zeros_like(yave)])
    

    def y_pred(self, xrc, sData):
        """
        Return images for each of the illumination patterns.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        pupilFn = self.mask*jnp.exp(1j*xrc[2])        
        for rxy in sData:
            pim = jnp.zeros(self.shape)
            for i in range(len(rxy)):
                pim += self.intensity(self.from_fourier(jnp.roll(xrcFT, rxy[i], (0,1)) * pupilFn))
            tmp.append(pim/float(len(rxy)))
        return jnp.array(tmp)
