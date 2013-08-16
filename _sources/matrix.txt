Matrix Distributions
======================

The *Distributions* package implements two matrix distributions: *Wishart* and *InverseWishart*.

Common Interface
------------------

Both distributions implement the same set of methods:

.. function:: mean(d)

	Return the mean matrix of d

.. function:: pdf(d, x)

	Compute the probability density at the input matrix x.

.. function:: logpdf(d, x)

	Compute the logarithm of the probability density at the input matrix x.

.. function:: rand(d)

	Draw a sample matrix from the distribution d.


Wishart Distribution
---------------------

The *Wishart distribution* is a multidimensional generalization of the Chi-square distribution, which is characterized by a degree of freedom ν, and a base matrix S.

.. code-block:: julia

	Wishart(nu, S)    # Wishart distribution with nu degrees of freedom and base matrix S.


Inverse-Wishart Distribution
------------------------------

The *Inverse-Wishart distribution* is usually used a the conjugate prior for the covariance matrix of a multivariate normal distribution, which is characterized by a degree of freedom ν, and a base matrix Φ.

.. code-block:: julia

	InverseWishart(nu, P)    # Inverse-Wishart distribution with nu degrees of freedom and base matrix P.
