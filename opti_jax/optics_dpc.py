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
import scipy

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
        
        
    def compute_loss_tv(self, x, Y, pats, lval):
        """
        Total variation loss function.
        """
        yPred = self.dpc_illumination(x, pats)
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness(x)*lval
        return loss
    
    
    def dpc_illumination(self, xrc, pats):
        """
        Return images for each of the DPC illumination patterns.
        """
        tmp = []
        for pat in pats:
            tmp.append(self.patterned_illumination(xrc, pat))
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
                    
        return [jnp.array(p0), jnp.array(p1), jnp.array(p2), jnp.array(p3)]


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
        

    def plot_patterns(self, pats, figsize = (5,5)):
        """
        Plot illumination patterns in k space.
        """
        fig, axs = plt.subplots(1, len(pats), figsize = (figsize[0]*len(pats), figsize[1]))

        for ii, pat in enumerate(pats):
            axs[ii].plot(self.dk0*pat[:,0], self.dk1*pat[:,1], "o", ms = 4)
            axs[ii].plot([-self.kmax, self.kmax], [0, 0], ":", color = "gray")
            axs[ii].plot([0, 0], [-self.kmax, self.kmax], ":", color = "gray")
        plt.show()


    def tv_smoothness(self, xrc):
        """
        Total variation, real and complex parts calculated independently.
        """
        t1 = jnp.mean(jnp.abs(jnp.diff(xrc[0], axis = 0))) + jnp.mean(jnp.abs(jnp.diff(xrc[0], axis = 1)))
        t2 = jnp.mean(jnp.abs(jnp.diff(xrc[1], axis = 0))) + jnp.mean(jnp.abs(jnp.diff(xrc[1], axis = 1)))
        return t1+t2


    def tv_solve(self, Y, pats, lval = 1.0e-3, learningRate = 1.0e-3, verbose = True):
        """
        Solve for best fit image with total variation regularization.
        """
        def fun(y, pats):
            return lambda x: self.compute_loss_tv(x, y, pats, lval)

        def stats(n, v):
            if verbose:
                print("{0:d} {1:.3e}".format(n, v))
            return [n, v]
            
        fn = jax.jit(fun(Y, pats))

        # Initialize
        yave = jnp.average(Y, axis = 0)
        x = jnp.array([yave, jnp.zeros_like(yave)])

        opt = optax.adam(learning_rate = learningRate)
        state = opt.init(x)

        n = 0
        nv = []
        for i in range(self.ni):
            for j in range(self.nj):
                v, g = jax.value_and_grad(fn)(x)
                if (n == 0):
                    nv.append(stats(n, v))
                    
                u, state = opt.update(g, state, x, value=v, grad=g, value_fn=fn)
                x = x + u

                # Clip to positive absorption values (x[0]).
                x = jnp.array([jnp.clip(x[0], 0.0, 1.0), x[1]])

                # Clip vector length to 1.0.
                rescale = 1.0/jnp.maximum(jnp.ones(x[0].shape), jnp.abs(x[0] + 1j*x[1]))
                x = jnp.array([x[0]*rescale, x[1]*rescale])

                n += 1

            nv.append(stats(n, v))

        return x, nv
    
