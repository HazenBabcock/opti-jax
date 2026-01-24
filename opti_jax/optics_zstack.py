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

        
    def compute_loss_tv_order1(self, x, Y, zdata, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.z_stack(x, zdata[0], zdata[1])
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order1(x)*lval
        return loss


    def compute_loss_tv_order2(self, x, Y, zdata, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.z_stack(x, zdata[0], zdata[1])
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order2(x)*lval
        return loss


    def make_bf_pattern(self, maxNA, ik0, ik1):
        """
        Approximate uniform illumination over an aperture with multiple point sources.
        
        ik0 and ik1 are expected to be integer arrays.
        """
        return super().make_bf_pattern(maxNA, ik0, ik1)
    
    
    def tv_solve(self, Y, zdata, lval = 1.0e-5, learningRate = 1.0e-1, order = 2, verbose = True):
        """
        Defaults tuned for a bright field focus stack.
        """
        return super().tv_solve(Y, [jnp.array(zdata[0]), jnp.array(zdata[1])], lval = lval, learningRate = learningRate, order = order, verbose = verbose)


    def x0(self, Y):
        """
        Assume middle image has the best focus.
        """
        ybest = Y[len(Y)//2]
        return jnp.array([ybest, jnp.zeros_like(ybest)])


    def z_stack(self, xrc, rxy, zvals):
        """
        Return images for each of the illumination z values.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for zv in zvals:
            zshift = jnp.exp(1j * 2.0 * jnp.pi * self.kz * zv)
            pim = jnp.zeros(self.shape)
            for rx, ry in rxy:
                pim += self.intensity(self.from_fourier(jnp.roll(xrcFT, (rx,ry), (0,1)) * zshift * self.mask))
            tmp.append(pim/float(len(rxy)))
        return jnp.array(tmp)


class OpticsZStackVP(OpticsZStack):
    """
    Z (focus) stack with variable pupil.
    """
    def __init__(self, pupilDelay = 50, **kwds):
        """
        Default is to start updating the pupil function after 50 iterations.
        """
        super().__init__(**kwds)        
        self.pupilDelay = pupilDelay


    def compute_loss_tv_order1(self, x, Y, zdata, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.z_stack(x, zdata[0], zdata[1])
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order1(x)*lval[0] + self.pupil_smoothness_order1_x(x[2])*lval[1]        
        return loss


    def compute_loss_tv_order2(self, x, Y, zdata, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.z_stack(x, zdata[0], zdata[1])
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order2(x)*lval[0] + self.pupil_smoothness_order2_x(x[2])*lval[1]
        return loss


    def make_bf_pattern(self, maxNA, ik0, ik1):
        """
        Approximate uniform illumination over an aperture with multiple point sources.
        
        ik0 and ik1 are expected to be integer arrays.
        """
        return super().make_bf_pattern(maxNA, ik0, ik1)

    
    def rescale(self, x, n):
        """
        Apply any constraints or corrections to x.
        """
        if (n > self.pupilDelay):
            return x
        else:
            return jnp.array([x[0], x[1], jnp.zeros_like(x[0])])

    
    def tv_solve(self, Y, zdata, lval = 1.0e-5, lvalp = 1.0e-2, learningRate = 1.0e-1, order = 2, verbose = True):
        """
        Defaults tuned for a bright field focus stack.
        """
        x, nv = super().tv_solve(Y, [jnp.array(zdata[0]), jnp.array(zdata[1])], lval = jnp.array([lval, lvalp]), learningRate = learningRate, order = order, verbose = verbose)
        x = jnp.mod(x + jnp.pi, 2*jnp.pi) - jnp.pi
        return x, nv
    

    def x0(self, Y):
        """
        Initialize w/ additional term for pupil function.
        """
        ybest = Y[len(Y)//2]
        return jnp.array([ybest, jnp.zeros_like(ybest), jnp.zeros_like(ybest)])
    
    
    def z_stack(self, xrc, rxy, zvals):
        """
        Return images for each of the illumination z values.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        pupilFn = self.mask*jnp.exp(1j*xrc[2])        
        for zv in zvals:
            zshift = jnp.exp(1j * 2.0 * jnp.pi * self.kz * zv)
            pim = jnp.zeros(self.shape)
            for rx, ry in rxy:
                pim += self.intensity(self.from_fourier(jnp.roll(xrcFT, (rx,ry), (0,1)) * zshift * pupilFn))
            tmp.append(pim/float(len(rxy)))
        return jnp.array(tmp)


class OpticsZStackVPMZ(OpticsZStackVP):
    """
    Z (focus) stack with variable pupil, solve at multiple z offsets.
    """
    def __init__(self, pupilDelay = 50, **kwds):
        """
        Default is to start updating the pupil function after 50 iterations.
        """
        super().__init__(**kwds)

        self.lastX = None
        

    def tv_solve(self, Y, zdata, zoffsets, lval = 1.0e-5, lvalp = 1.0e-2, learningRate = 1.0e-1, order = 2, verbose = True):
        """
        Solve at multiple z offsets.
        """
        self.lastX = None
        
        xs = []
        nvs = []
        for zo in zoffsets:
            if verbose:
                print("Solving at z offset", zo)
                
            x, nv = super().tv_solve(Y, [zdata[0], zdata[1] + zo], lval = lval, lvalp = lvalp, learningRate = learningRate, order = order, verbose = verbose)
            self.lastX = x
            xs.append(x)
            nvs.append(nvs)
            if verbose:
                print()

        return xs, nvs


    def x0(self, Y):
        """
        Initialize w/ additional term for pupil function.
        """
        if self.lastX is None:
            ybest = Y[len(Y)//2]
            return jnp.array([ybest, jnp.zeros_like(ybest), jnp.zeros_like(ybest)])
        else:
            return jnp.copy(self.lastX)
