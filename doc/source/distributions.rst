Distribution Types
===================

Each distribution is implemented by a sub-type of ``Distribution``. ``Distribution`` is an abstract type declared as 

.. code-block:: julia

    abstract Distribution{F<:VariateForm,S<:ValueSupport}

The ``Distribution`` type has two parameters: 

- ``F`` specifies the form of the variate, which can be either of the following:

    * ``Univariate``:  each sample is a scalar
    * ``Multivariate``: each sample is a vector
    * ``Matrixvariate``: each sample is a matrix

- ``S`` specifies the support of sample elements, which can be either of the following:

    * ``Continuous``: samples take continuous real values
    * ``Discrete``: samples take discrete values (*e.g.* integers)

For example, a continuous univariate distribution is an instance of ``Distribution{Univariate,Continuous}``.

To simplify the use in practice, we introduce a series of type alias as follows:

.. code-block:: julia

    typealias UnivariateDistribution{S<:ValueSupport}   Distribution{Univariate,S}
    typealias MultivariateDistribution{S<:ValueSupport} Distribution{Multivariate,S}
    typealias MatrixDistribution{S<:ValueSupport}       Distribution{Matrixvariate,S}
    typealias NonMatrixDistribution Union(UnivariateDistribution, MultivariateDistribution)

    typealias DiscreteDistribution{F<:VariateForm}   Distribution{F,Discrete}
    typealias ContinuousDistribution{F<:VariateForm} Distribution{F,Continuous}

    typealias DiscreteUnivariateDistribution     Distribution{Univariate,    Discrete}
    typealias ContinuousUnivariateDistribution   Distribution{Univariate,    Continuous}
    typealias DiscreteMultivariateDistribution   Distribution{Multivariate,  Discrete}
    typealias ContinuousMultivariateDistribution Distribution{Multivariate,  Continuous}
    typealias DiscreteMatrixDistribution         Distribution{Matrixvariate, Discrete}
    typealias ContinuousMatrixDistribution       Distribution{Matrixvariate, Continuous}

Generally, the distribution type determines the type of the samples:

* Each sample from a univariate distribution is a scalar.
* Each sample from a multivariate distribution is a vector.
* Each sample from a matrix distribution is a matrix.
* The element type of a sample from a discrete distribution is ``Int``.
* The element type of a sample from a continuous distribution is ``Float64``.
