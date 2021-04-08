# manga-dupl

When we take a spectrum, we expose a sensor to light, and then correct the measured photon counts per wavelength channel according to a calibration curve. The calibration curve itself changes, based on observing conditions, and the data reduction process merely infers it by comparison between observations of single stars and those stars' known spectra. This process is called spectrophotometric caliabration.

Spectrophotometric calibration is known to be somewhat uncertain, and (in the case of the MaNGA survey, at least), the uncertainty is covariate between wavelength channels. The object of this brief study is to infer this covariance structure by comparing the spectra of objects that have been observed multiple times. Each re-observation takes place under slightly different conditions, and so by subtracting two theoretically-equal observations, we obtain one realization of the (covariate) spectrophotometric error. In a sufficiently large survey, the number of duplicate observations will be large enough to build a complete view of the uncertainty induced by spectrophotometric calibration.

At a high level, these are the necessary steps:

- Find single galaxies with multiple observations
- Match locations in the datacube which correspond to equivalent spectra
- Subtract equivalent spectra
- Find a smooth representation of the covariance that results in the residuals from subtracting theoretically-identical spectra

## Smoothness
A smooth representation of the covariance can be obtained by recognizing the relationship between principal component analysis (PCA) and the covariance. That is, if we can obtain a smooth principal component basis set representing the subtracted spectra, the associated covariance matrix will also be smooth. While the technique of functional PCA provides an overall philosophy, the current software solutions available (at least in python) can't really handle the huge vectors that optical spectroscopy relies upon (and while there might be sparsity arguments to be made, we don't have a physical model to inform a specific choice of sparsity).

The alternative is to allow the data to guide their own smooth representation. Specifically, residuals between two observations of the same object ought to have a reasonably coherent smoothness to them. This allows us to first infer the smoothness on smaller groups of data, and then build up a principal component system corresponding to the whole dataset.

I chose to smooth the spectroscopic residuals for one object at a time (each object contains many lines-of-sight, each of which has multiple observations) using B-splines, then use the B-spline fits to refine an Incremental PCA model. In high dimension, this is a slow, but steady process, and produces realistic, smooth covariances.
