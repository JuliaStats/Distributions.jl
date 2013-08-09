Multivariate Distributions
=============================

*Multivariate distributions* are probabilistic distributions whose samples are vectors. The *Distributions* package implements several commonly used multivariate distributions, including *Multinomial*, *Multivairate Normal* and *Dirichlet*.

Multinomial Distribution
---------------------------

The *multinomial distribution* generalizes the *binomial distribution*. Consider n independent draws from a Categorical distribution over a finite set of size k, and let :math:`X = (X_1, ..., X_k)` where ``X_i`` represents the number of times the element ``i`` occurs, then the distribution of ``X`` is a multinomial distribution. Each sample of a multinomial distribution is a k-dimensional integer vector that sums to n.

The probability mass function is given by

.. math::

    f(x; n, p) = \frac{n!}{x_1! \cdots x_k!} \prod_{i=1}^k p_i^{x_i}, 
    \quad x_1 + \cdots + x_k = n

.. code-block:: julia

    Multinomial(n, p)   # Multinomial distribution for n trials with probability vector p


Multivariate Normal Distribution
----------------------------------

The *multivariate normal distribution* is a multidimensional generalization of the *normal distribution*. The probability density function of a d-dimensional multivariate normal distribution with mean vector μ and covariance matrix Σ is 

.. math::

    f(x; \mu, \Sigma) = \frac{1}{(2 \pi)^{d/2} |\Sigma|^{1/2}}
    \exp \left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)


.. code-block:: julia

    # Let sig be a scalar
    MultivariateNormal(mu, sig)     # Multivariate normal distribution 
                                    # with mean mu and covariance sig^2 * I

    # Let v be a vector
    MultivariateNormal(mu, v)       # Multivariate normal distribution
                                    # with mean mu and covariance diagm(v)

    # Let C be a positive definite matrix
    MultivariateNormal(mu, C)       # Multivariate normal distribution
                                    # with mean mu and covariance C

To save some typing, we introduce the name ``MvNormal`` as an alias of ``MultivariateNormal``. One can use ``MvNormal`` in the place of ``MultivariateNormal`` (*e.g.* ``MvNormal(mu, C)``.)


The MultivariateNormal type and specialized covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*MultivariateNormal* is a parameteric type declared as below, which takes ``Cov``, the type of the covariance, as an argument, thus enables specialized handling of covariance matrices of different structures.

.. code-block:: julia

    immutable MultivariateNormal{Cov<:AbstractPDMat} <: ContinuousMultivariateDistribution

In particular, ``ScalMat``, ``PDiagMat`` and ``PDMat`` are respectively used as the covariance type for cases where the covariance matrix is respectively isotropic, diagonal, or a full matrix. Consequently, more efficient implementation will be internally used when handling covariance matrix with special structures.


Dirichlet Distribution
------------------------

The Dirichlet distribution is often used the conjugate prior for Categorical or Multinomial distributions. The probability density function of a Dirichlet distribution with parameter :math:`\alpha = (\alpha_1, \ldots, \alpha_k)` is

.. math::

    f(x; \alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^k x_i^{\alpha_i - 1}, \quad \text{ with }
    B(\alpha) = \frac{\prod_{i=1}^k \Gamma(\alpha_i)}{\Gamma \left( \sum_{i=1}^k \alpha_i \right)}, 
    \quad x_1 + \cdots + x_k = 1


.. code-block:: julia

    # Let alpha be a vector
    Dirichlet(alpha)         # Dirichlet distribution with parameter vector alpha

    # Let a be a positive scalar
    Dirichlet(k, a)          # Dirichlet distribution with parameter a * ones(k)  






