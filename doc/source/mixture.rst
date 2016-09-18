.. _mixture:

Mixture Models
=================

A `mixture model <http://en.wikipedia.org/wiki/Mixture_model>`_ is a probabilistic distribution that combines a set of *component* to represent the overall distribution. Generally, the probability density/mass function is given by a convex combination of the pdf/pmf of individual components, as

.. math::

    f_{mix}(x; \Theta, \pi) = \sum_{k=1}^K \pi_k f(x; \theta_k)

A *mixture model* is characterized by a set of component parameters :math:`\Theta=\{\theta_1, \ldots, \theta_K\}` and a prior distribution :math:`\pi` over these components. 


Type Hierarchy
-----------------

This package introduces a type ``MixtureModel``, defined as follows, to represent a *mixture model*:

.. code-block:: julia

    abstract AbstractMixtureModel{VF<:VariateForm,VS<:ValueSupport} <: Distribution{VF, VS}

    immutable MixtureModel{VF<:VariateForm,VS<:ValueSupport,Component<:Distribution} <: AbstractMixtureModel{VF,VS}
        components::Vector{Component}
        prior::Categorical
    end

    typealias UnivariateMixture    AbstractMixtureModel{Univariate} 
    typealias MultivariateMixture  AbstractMixtureModel{Multivariate}

**Remarks:**

- We introduce ``AbstractMixtureModel`` as a base type, which allows one to define a mixture model with different internal implementation, while still being able to leverage the common methods defined for ``AbstractMixtureModel``.

- The ``MixtureModel`` is a parametric type, with three type parameters: 

    - ``VF``: the variate form, which can be ``Univariate``, ``Multivariate``, or ``Matrixvariate``.
    - ``VS``: the value support, which can be ``Continuous`` or ``Discrete``.
    - ``Component``: the type of component distributions, *e.g.* ``Normal``.

- We define two aliases: ``UnivariateMixture`` and ``MultivariateMixture``.

With such a type system, the type for a mixture of univariate normal distributions can be written as 

.. code-block:: julia

    MixtureModel{Univariate,Continuous,Normal}


Construction
--------------

A mixture model can be constructed using the constructor ``MixtureModel``. Particularly, we provide various methods to simplify the construction.

.. function:: MixtureModel(components, prior)

    Construct a mixture model with a vector of components and a prior probability vector.

.. function:: MixtureModel(components)

    Construct a mixture model with a vector of components. All components share the same prior probability.

.. function:: MixtureModel(C, params, prior)

    Construct a mixture model with component type ``C``, a vector of parameters for constructing the components given by ``params``, and a prior probability vector.

.. function::  MixtureModel(C, params)

    Construct a mixture model with component type ``C`` and a vector of parameters for constructing the components given by ``params``. All components share the same prior probability.


**Examples**

.. code-block:: julia

    # constructs a mixture of three normal distributions,
    # with prior probabilities [0.2, 0.5, 0.3]
    MixtureModel(Normal[
       Normal(-2.0, 1.2),
       Normal(0.0, 1.0),
       Normal(3.0, 2.5)], [0.2, 0.5, 0.3])

    # if the components share the same prior, the prior vector can be omitted
    MixtureModel(Normal[
       Normal(-2.0, 1.2),
       Normal(0.0, 1.0),
       Normal(3.0, 2.5)])

    # Since all components have the same type, we can use a simplified syntax
    MixtureModel(Normal, [(-2.0, 1.2), (0.0, 1.0), (3.0, 2.5)], [0.2, 0.5, 0.3])

    # Again, one can omit the prior vector when all components share the same prior
    MixtureModel(Normal, [(-2.0, 1.2), (0.0, 1.0), (3.0, 2.5)])

    # The following example shows how one can make a Gaussian mixture
    # where all components share the same unit variance 
    MixtureModel(map(u -> Normal(u, 1.0), [-2.0, 0.0, 3.0]))


Common Interface
------------------

All subtypes of ``AbstractMixtureModel`` (obviously including ``MixtureModel``) provide the following two methods:

.. function:: components(d)

    Get a list of components of the mixture model ``d``.

.. function:: probs(d)

    Get the vector of prior probabilities of all components of ``d``.

.. function:: component_type(d)

    The type of the components of ``d``.


In addition, for all subtypes of ``UnivariateMixture`` and ``MultivariateMixture``, the following generic methods are provided:

.. function:: mean(d)

    Compute the overall mean (expectation).

.. function:: var(d)

    Compute the overall variance (only for ``UnivariateMixture``).

.. function:: length(d)

    The length of each sample (only for ``Multivariate``).

.. function:: pdf(d, x)

    Evaluate the (mixed) probability density function over ``x``. Here, ``x`` can be a single sample or an array of multiple samples.

.. function:: logpdf(d, x)

    Evaluate the logarithm of the (mixed) probability density function over ``x``. Here, ``x`` can be a single sample or an array of multiple samples.

.. function:: rand(d)

    Draw a sample from the mixture model ``d``.

.. function:: rand(d, n)

    Draw ``n`` samples from ``d``.

.. function:: rand!(d, r)

    Draw multiple samples from ``d`` and write them to ``r``.


Estimation
-----------

There are a number of methods for estimating of mixture models from data, and this problem remains an open research topic. This package does not provide facilities for estimaing mixture models. One can resort to other packages, *e.g.* *GaussianMixtures.jl*, for this purpose.


