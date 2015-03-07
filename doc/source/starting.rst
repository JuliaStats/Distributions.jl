Getting Started
===============

Installation
-----------
The Distributions package is available through the Julia package system by running ``Pkg.add("Distributions")``. Throughout, we assume that you have installed the package.

Starting With a Normal Distribution
-----------

We start by drawing :math:`100` observations from a standard-normal random variable. After setting up the environment through

.. code-block:: julia

    julia> using Distributions
    julia> srand(123) # Setting the seed

We first create a ``normal`` distribution and then obtain samples using ``rand``:

.. code-block:: julia

    julia> normal = Normal()
    Normal(μ=0.0, σ=1.0)

    julia> x = rand(normal, 100)
    100-element Array{Float64,1}:
      0.376264
     -0.405272
     ...

You can easily obtain a pdf, cdf, percentile, and many other functions for a distribution. For instance, the median (50th percentile) and the 95th percentile for the standard-normal distribution are

.. code-block:: julia

    julia> quantile(Normal(), [0.5, 0.95])
    2-element Array{Float64,1}:
     0.0
     1.64485

The normal distribution is parameterized by a mean (location) and standard deviation (scale). To draw random samples from a normal distribution with mean 1 and standard deviation 2:

.. code-block:: julia

    julia> rand(Normal(1, 2), 100)

Using Other Distributions
-----------

The package contains a large number of additional distributions of four main types:

* ``Univariate``
* ``Truncated``
* ``Multivariate``
* ``Matrixvariate``

The ``Univariate`` random variables split further into ``Discrete`` and ``Continuous``.

For instance, you can define the following distributions (among many others):

.. code-block:: julia

    julia> Binomial(p) # Discrete univariate
    julia> Cauchy(u, b)  # Continuous univariate
    julia> TruncatedNormal(Normal(mu, sigma), l, u) # Truncated
    julia> Multinomial(n, p) # Multivariate
    julia> Wishart(nu, S) # Matrix-variate

To find out which parameters are appropriate for a given distribution ``D``, you can use ``names(D)``:

.. code-block:: julia

    julia> names(Cauchy)
    2-element Array{Symbol,1}:
     :μ
     :β

This tells you that a Cauchy distribution is initialized with location ``μ`` and scale ``β``.

Estimate the Parameters
------------------

It is often useful to approximate an empirical distribution with a theoretical distribution. As an example, we can use the array ``x`` we created above and ask which normal distribution best describes it:

.. code-block:: julia

    julia> fit(Normal, x)
    Normal(μ=0.036692077201688635, σ=1.1228280164716382)

Since ``x`` is a random draw from ``Normal``, it's easy to check that the fitted values are sensible. Indeed, the estimates :math:`[0.04, 1.12]` are close to the true values of :math:`[0.0, 1.0]` that we used to generate ``x``.

Create Mixture Models
------------------

Creating mixture models is simple. For instance, you can create a mixture of three normal variables with prior probabilities :math:`0.2, 0.5, 0.3` as follows:

.. code-block:: julia

    julia> m = MixtureModel(Normal[
                   Normal(-2.0, 1.2),
                   Normal(0.0, 1.0),
                   Normal(3.0, 2.5)], [0.2, 0.5, 0.3])

A mixture model can be accessed using a smaller set of functions than the pre-defined distributions. While a pdf is defined:

.. code-block:: julia

    julia> pdf(m, 2)
    0.07144494659237469

a quantile is not defined.

This package does not provide facilities for estimating mixture models. One can resort to other packages, *e.g.* *MixtureModels.jl*, for this purpose.