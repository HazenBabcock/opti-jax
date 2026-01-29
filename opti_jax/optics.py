#!/usr/bin/env python
"""
Optics base class.

Hazen 2026.01
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy


class Optics(object):
    """
    Base class for JAX optics.
    """
    def __init__(self, NA = None, NI = 1.0, pixelSize = None, shape = None, wavelength = None, **kwds):
        """
        NA - objective numerical aperature.
        NI - refractive index of immersion media.
        pixelSixe - pixel size in microns.
        shape - image dimensions in pixes.
        wavelength - wavelength in microns.
        """
        super().__init__(**kwds)

        self.NA = NA
        self.NI = NI
        self.pixelSize = pixelSize
        self.shape = shape
        self.wavelength = wavelength

        self.kmax = self.NA/self.wavelength

        self.dk0 = 1.0/(self.shape[0] * self.pixelSize)
        self.dk1 = 1.0/(self.shape[1] * self.pixelSize)
        self.rmax0 = self.kmax/self.dk0
        self.rmax1 = self.kmax/self.dk1

        [self.g0,self.g1] = np.mgrid[-self.shape[0]//2 : self.shape[0]//2, -self.shape[1]//2 : self.shape[1]//2]

        k0 = self.dk0 * self.g0
        k1 = self.dk1 * self.g1
        self.k = np.sqrt(k0 * k0 + k1 * k1)

        tmp = self.NI/self.wavelength
        self.kz = np.lib.scimath.sqrt(tmp * tmp - self.k * self.k)
        self.kz[(self.k > tmp)] = 0.0

        # Mask for maximum pass frequency.
        self.r = self.k/self.kmax
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


    def plot_x(self, x, figsize = (8, 8)):
        """
        Plot values of x in the complex plane.
        """
        fig, axs = plt.subplots(1, 1, figsize = figsize)

        axs.plot(x[0].flatten(), x[1].flatten(), ".", alpha = 0.01)
        axs.plot([-1.2, 1.2], [0.0, 0.0], ":", color = "gray")
        axs.plot([0.0, 0.0], [-1.2, 1.2], ":", color = "gray")
        axs.set_xlim(-1.1, 1.1)
        axs.set_ylim(-1.1, 1.1)
        axs.set_xlabel("Real")
        axs.set_ylabel("Imag")
        plt.show()
        

#    def plot_x(self, x, figsize = (4.75, 8)):
#        """
#        Plot values of x in the complex plane.
#        """
#        fig, axs = plt.subplots(1, 1, figsize = figsize)
#
#        axs.plot(x[0].flatten(), x[1].flatten(), ".", alpha = 0.01)
#        axs.plot([-0.2, 1.2], [0.0, 0.0], ":", color = "gray")
#        axs.plot([0.0, 0.0], [-1.2, 1.2], ":", color = "gray")
#        axs.set_xlim(-0.1, 1.1)
#        axs.set_ylim(-1.1, 1.1)
#        axs.set_xlabel("Real")
#        axs.set_ylabel("Imag")
#        plt.show()        
        

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
        Approximate uniform illumination over an aperture with multiple point sources.
        
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


    def patterned_illumination(self, xrc, rxy):
        """
        xrc is [real, complex] as not all Optax solvers handle complex numbers.
        """
        return self.patterned_illumination_ft(self.to_fourier(self.illuminate(xrc, 0.0, 0.0)), rxy)

    
    def patterned_illumination_ft(self, xrcFT, rxy):
        """
        For efficiency we roll the fourier transform instead of recalculating the illumination for all k values.
        For efficiency we roll the mask, sum and then do the inverse fourier transform.

        Note that this is an approximation, the resulting image is slightly different from
        patterned_illumination_no_roll_ft().
        """
        pim = jnp.zeros_like(xrcFT)
        for rx, ry in rxy:
            pim += jnp.roll(self.mask, (-rx,-ry), (0,1)) * xrcFT
        return self.intensity(self.from_fourier(pim)/float(len(rxy)))

    
    def patterned_illumination_no_roll(self, xrc, rxy):
        """
        xrc is [real, complex] as not all Optax solvers handle complex numbers.
        """
        return self.patterned_illumination_no_roll_ft(self.to_fourier(self.illuminate(xrc, 0.0, 0.0)), rxy)

    
    def patterned_illumination_no_roll_ft(self, xrcFT, rxy):
        """
        For efficiency we roll the fourier transform instead of recalculating the illumination for all k values.
        This versions performs the inverse fourier transform for each illumination angle then adds the results.
        """
        pim = jnp.zeros(xrcFT.shape)
        for rx, ry in rxy:
            pim = pim + self.intensity(self.from_fourier(jnp.roll(xrcFT, (rx,ry), (0,1))*self.mask))
        return pim/float(len(rxy))


    def plot_pattern(self, pat, figsize = (5,5)):
        """
        Plot illumination pattern in k space.
        """
        fig, axs = plt.subplots(1, 1, figsize = figsize)
        axs.plot(self.dk0*pat[:,0], self.dk1*pat[:,1], "o", ms = 4)
        axs.plot([-1.1*self.kmax, 1.1*self.kmax], [0, 0], ":", color = "gray")
        axs.plot([0, 0], [-1.1*self.kmax, 1.1*self.kmax], ":", color = "gray")

        # Objective kmax.
        circ = plt.Circle((0.0, 0.0), radius = self.kmax, edgecolor="green", facecolor = "none", linestyle = ":")
        axs.add_patch(circ)

        # Absolute kmax (for air objective)?
        #circ = plt.Circle((0.0, 0.0), radius = 0.5*jnp.pi, edgecolor="black", facecolor = "none", linestyle = ":")
        #axs.add_patch(circ)
        
        plt.show()


    def plot_stats(self, stats):
        """
        Plot solver convergence array values.
        """
        statsNp = np.array(stats)
        print(statsNp.shape)
        
        fig, axs = plt.subplots(2, 1, figsize = (8, 8))
        axs[0].plot(statsNp[:,0], statsNp[:,1])
        axs[0].set_ylabel("Loss")
        axs[0].set_xlabel("Iterations")
        
        axs[1].plot(statsNp[:,0], statsNp[:,1])
        axs[1].set_ylabel("Loss")
        axs[1].set_xlabel("Iterations")
        axs[1].semilogy()
        
        plt.show()


    def pupil_smoothness_order1_x(self, p):
        """
        First order total variation in pupil function.
        """
        return jnp.mean(self.mask*(jnp.abs(p - jnp.roll(p, 1, axis = 0)))) + jnp.mean(self.mask*(jnp.abs(p - jnp.roll(p, 1, axis = 1))))
    

    def pupil_smoothness_order2_x(self, p):
        """
        Second order total variation in pupil function.
        """
        tv = jnp.mean(self.mask*(jnp.abs(2*p - jnp.roll(p, 1, axis = 0) - jnp.roll(p, -1, axis = 0))))
        return tv + jnp.mean(self.mask*(jnp.abs(2*p - jnp.roll(p, 1, axis = 1) - jnp.roll(p, -1, axis = 1))))


    def rescale(self, x, n):
        """
        Clip maximum reconstructed image magnitude to 1.0.
        """
        return x
        #rescale = 1.0/jnp.maximum(jnp.ones(x[0].shape), jnp.abs(x[0] + 1j*x[1]))
        #return jnp.array([x[0]*rescale, x[1]*rescale])


    def tv_smoothness_order1(self, xrc):
        """
        First order total variation in X/Y, real and complex parts calculated independently.
        """
        return self.tv_smoothness_order1_x(xrc[0]) + self.tv_smoothness_order1_x(xrc[1])


    def tv_smoothness_order1_x(self, x):
        """
        First order total variation in X/Y.
        """
        return jnp.mean(jnp.abs(x - jnp.roll(x, 1, axis = 0))) + jnp.mean(jnp.abs(x - jnp.roll(x, 1, axis = 1)))

    
    def tv_smoothness_order2(self, xrc):
        """
        Second order total variation in X/Y, real and complex parts calculated independently.
        """
        return self.tv_smoothness_order2_x(xrc[0]) + self.tv_smoothness_order2_x(xrc[1])

    
    def tv_smoothness_order2_x(self, x):
        """
        Second order total variation in X/Y.
        """
        tv = jnp.mean(jnp.abs(2*x - jnp.roll(x, 1, axis = 0) - jnp.roll(x, -1, axis = 0)))
        return tv + jnp.mean(jnp.abs(2*x - jnp.roll(x, 1, axis = 1) - jnp.roll(x, -1, axis = 1)))
    

    def tv_solve(self, Y, illm, lval = 1.0e-3, learningRate = 1.0e-3, order = 1, verbose = True):
        """
        Solve for best fit image with total variation regularization.

        The images to fit (Y) should be normalized to have intensities in the range 0.0 - 1.0.
        """
        def fun1(y, illm):
            return lambda x: self.compute_loss_tv_order1(x, y, illm, lval)

        def fun2(y, illm):
            return lambda x: self.compute_loss_tv_order2(x, y, illm, lval)

        def stats(n, v):
            if verbose:
                print("{0:d} {1:.3e}".format(n, v))
            return [int(n), float(v)]

        if (order == 1):
            fn = jax.jit(fun1(Y, illm))
        elif (order == 2):
            fn = jax.jit(fun2(Y, illm))
        else:
            assert False, f"Order {order} not available"
            
        # Initialize
        x = self.x0(Y)
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
                x = self.rescale(x + u, n)
                
                n += 1

            nv.append(stats(n, v))

        return x, nv


    def x0(self, Y):
        yave = jnp.average(Y, axis = 0)
        return jnp.array([yave, jnp.zeros_like(yave)])
