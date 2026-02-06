#!/usr/bin/env python
"""
PtyLab HDF5 file format IO.

Hazen 2026.02
"""

import h5py
import numpy as np


def load_ptylab_hdf5(filename):
    """
    Load all the pages in a tiff stack (assumed to be the same size and type).
    """
    data = {}
    with h5py.File(filename) as h5f:
        flds = h5f.keys()
        for key in ["NA", "dxd", "magnification", "wavelength", "zled"]:
            if key in flds:
                data[key] = float(h5f[key][()])

        # Unit conversion (meters -> microns).
        data["pixelSize"] = 1.0e6 * data["dxd"]/data["magnification"]
        data["wavelength"] = 1.0e6 * data["wavelength"]

        for key in ["encoder", "ptychogram"]:
            data[key] = np.array(h5f[key])

        # Calculate encoder k space values.
        exy = data["encoder"]
        data["k01"] = -(exy / np.sqrt(exy[:,0]**2 + exy[:,1]**2 + data["zled"]**2)[:,None])

        # Calculate a normalized ptychogram.
        data["Y"] = 0.8*data["ptychogram"]/np.max(data["ptychogram"])
        
    return data
