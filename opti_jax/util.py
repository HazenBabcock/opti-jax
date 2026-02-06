#!/usr/bin/env python
"""
Utility functions.

Hazen 2026.01
"""

import numpy as np
import scipy
import tifffile


def kvec_to_r01(kvec, ozs):
    """
    Convert k space vectors to integer shift values.
    """
    r0 = kvec[:,0]/ozs.dk0/ozs.wavelength
    r1 = kvec[:,1]/ozs.dk1/ozs.wavelength
    return np.round(np.column_stack((r0,r1))).astype(int)


def load_stack(sname):
    """
    Load all the pages in a tiff stack (assumed to be the same size and type).
    """
    stk = []
    with tifffile.TiffFile(sname) as tfr:
        for page in tfr.pages:
            stk.append(page.asarray())
    return np.array(stk)


