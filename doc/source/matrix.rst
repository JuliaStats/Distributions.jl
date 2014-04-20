Matrix Distributions
======================

The *Distributions* package implements two matrix distributions: *Wishart* and *InverseWishart*.

Common Interface
------------------

Both distributions implement the same set of methods:

- **mean** (d)

	Return the mean matrix of ``d``.

- **pdf** (d, x)

	Compute the probability density at the input matrix ``x``.

- **logpdf** (d, x)

	Compute the logarithm of the probability density at the input matrix ``x``.

- **rand** (d)

	Draw a sample matrix from the distribution ``d``.


Variate Dimensions
~~~~~~~~~~~~~~~~~~~~

The functions ``size`` and ``length`` apply to any type of distribution.

- **size** (d)

   ``size(d)==size(rand(d))``, that is, the size of the samples from this distribution.

- **length** (d)

   ``length(d)==length(rand(d))``, that is, the total number of elements in a sampled matrix.


Wishart Distribution
---------------------

The `Wishart distribution <http://en.wikipedia.org/wiki/Wishart_distribution>`_ is a multidimensional generalization of the Chi-square distribution, which is characterized by a degree of freedom ν, and a base matrix S.

.. code-block:: julia

	Wishart(nu, S)    # Wishart distribution with nu degrees of freedom and base matrix S.


Inverse-Wishart Distribution
------------------------------

The `Inverse Wishart distribution <http://en.wikipedia.org/wiki/Inverse-Wishart_distribution>`_ is usually used a the conjugate prior for the covariance matrix of a multivariate normal distribution, which is characterized by a degree of freedom ν, and a base matrix Φ.

.. code-block:: julia

	InverseWishart(nu, P)    # Inverse-Wishart distribution with nu degrees of freedom and base matrix P.
