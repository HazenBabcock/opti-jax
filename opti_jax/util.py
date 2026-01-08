#!/usr/bin/env python
"""
Utility functions.

Hazen 2026.01
"""

import numpy as np
import scipy
import tifffile


def load_stack(sname):
    """
    Load all the pages in a tiff stack (assumed to be the same size and type).
    """
    stk = []
    with tifffile.TiffFile(sname) as tfr:
        for page in tfr.pages:
            stk.append(page.asarray())
    return np.array(stk)


