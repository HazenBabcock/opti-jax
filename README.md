## Opti-Jax ##

(Scalar) optics using Jax and Optax.

This package uses scalar optics theory and Jax for the forward model for several brightfield imaging variations including differential phase contrast / quantative phase microscopy and fourier phytograpy. Jax's autodiff and solvers from Optax are used to solve the inverse problem, recovering the complex object that best fits the observed intensity measurements.

### Install ###

#### Source ####

```
$ git https://github.com/HazenBabcock/opti-jax.git
$ cd opti-jax
$ python -m pip install .
```

### Usage ###

Please see the [example](https://github.com/HazenBabcock/opti-jax/tree/main/notebooks) Jupyter notebooks.
