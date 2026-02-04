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
import skimage

import opti_jax.optics as optics


class OpticsFPty(optics.OpticsBF):
    """
    Fourier Phytography illumination.
    """
    def __init__(self, fitShape = None, ni = 20, nj = 50, **kwds):
        """
        fitShape - size of final pty image, this should be multiple of shape.
        pixelSize - this is the pixel size in the FP images, not the final pty image.
        shape - this is the size of the FP images.
        
        Default is 1000 iterations for solving (ni * nj).
        """
        shape = kwds["shape"]
        
        scf0 = float(fitShape[0])/float(shape[0])
        scf1 = float(fitShape[1])/float(shape[1])
        assert (np.abs(scf0 - scf1)/(scf0 + scf1) < 1.0e-6), "'fitShape' dimensions must be proportional to 'shape' dimensions."
        kwds["pixelSize"] = kwds["pixelSize"]/scf0
        kwds["shape"] = fitShape
        
        super().__init__(**kwds)

        self.ni = ni
        self.nj = nj
        
        # For slicing out the center of the HR image in Fourier space.
        c0 = fitShape[0]//2
        c1 = fitShape[1]//2
        hw0 = shape[0]//2
        hw1 = shape[1]//2
        self.slicer = (slice(c0-hw0,c0+hw0),slice(c1-hw1,c0+hw1))

        # Scaling factor for fourier transform.
        self.ftsc = 1.0/(scf0*scf0)

        # Weighting factors for different images.
        self.weights = None
        

    def calculate_weights(self, Y):
        """
        Return weight values to use for different FT images.
        """
        means = []
        for im in Y:
            means.append(jnp.mean(im[im>0.0]))
            
        return 1.0/(jnp.array(means))


    def compute_loss_tv_order1(self, x, Y, sData, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.y_pred(x, sData)
        loss = jnp.mean(optax.l2_loss(yPred, Y)*self.weights[:,None,None]) + self.tv_smoothness_order1(x)*lval
        return loss


    def compute_loss_tv_order2(self, x, Y, sData, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.y_pred(x, sData)
        loss = jnp.mean(optax.l2_loss(yPred, Y)*self.weights[:,None,None]) + self.tv_smoothness_order2(x)*lval
        return loss


    def estimate_intensities(self, Y, yPred):
        """
        Estimate intensity correction factors for different illuminations.
        """
        ypm = jnp.mean(yPred, axis = (1,2))
        return 1.0 - jnp.mean((Y - yPred), axis = (1,2))/ypm

    
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

        # Object size in fourier space.
        hw0 = self.dk0*(self.shape[0]/2)
        hw1 = self.dk1*(self.shape[1]/2)
        rect = plt.Rectangle((-hw0, -hw1), 2*hw0, 2*hw1, edgecolor="red", facecolor = "none", linestyle = ":")
        axs.add_patch(rect)

        plt.show()

        
    def solve_tv(self, Y, sData, lval = 1.0e-3, learningRate = 1.0e-1, optimizer = None, order = 2, verbose = True, weights = None, x0 = None):
        """
        Defaults tuned for fourier phytography.
        """
        if weights is None:
            self.weights = self.calculate_weights(Y)
        else:
            self.weights = jnp.copy(jnp.array(weights))
            
        return super().solve_tv(Y, sData,
                                lval = lval,
                                learningRate = learningRate,
                                optimizer = optimizer,
                                order = order,
                                verbose = verbose,
                                x0 = x0)


    def x0(self, Y):
        yave = skimage.transform.resize(jnp.average(Y, axis = 0), self.shape, preserve_range = True)
        return jnp.array([yave, jnp.zeros_like(yave)])

    
    def y_pred(self, xrc, sData):
        """
        Return images for each of the illumination 'angles'.

        rxy is the (integer) shift value to use in k space for each illumination angle.
        """
        [rxy, intensities] = sData

        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for i in range(len(rxy)):
            tmp.append(self.intensity(self.from_fourier(self.ftsc*(jnp.roll(xrcFT, rxy[i], (0,1)) * self.mask)[self.slicer])) * intensities[i])
        return jnp.array(tmp)

    
class OpticsFPtyVP(optics.OpticsBFVP, OpticsFPty):
    """
    Fourier Phytography illumination with variable pupil.
    """
    def solve_tv(self, Y, sData, lval = 1.0e-3, lvalp = 1.0e-2, learningRate = 1.0e-1, optimizer = None, order = 2, verbose = True, weights = None, x0 = None):
        """
        Defaults tuned for fourier phytography.
        """
        if weights is None:
            self.weights = self.calculate_weights(Y)
        else:
            self.weights = jnp.copy(jnp.array(weights))

        x, nv = super().solve_tv(Y, sData,
                                 lval = jnp.array([lval, lvalp]),
                                 learningRate = learningRate,
                                 optimizer = optimizer,
                                 order = order,
                                 verbose = verbose,
                                 x0 = x0)
        
        x = jnp.mod(x + jnp.pi, 2*jnp.pi) - jnp.pi
        return x, nv


    def x0(self, Y):
        """
        Initialize w/ additional term for pupil function.
        """
        yave = skimage.transform.resize(jnp.average(Y, axis = 0), self.shape, preserve_range = True)
        return jnp.array([yave, jnp.zeros_like(yave), jnp.zeros_like(yave)])


    def y_pred(self, xrc, sData):
        """
        Return images for each of the illumination 'angles'.

        rxy is the (integer) shift value to use in k space for each illumination angle.
        """
        [rxy, intensities] = sData
        
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        pupilFn = self.mask*jnp.exp(1j*xrc[2])
        for i in range(len(rxy)):
            tmp.append(self.intensity(self.from_fourier(self.ftsc*(jnp.roll(xrcFT, (rx,ry), (0,1)) * pupilFn)[self.slicer])) * intensities[i])
        return jnp.array(tmp)
