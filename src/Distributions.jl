module Distributions

using NumericExtensions
using Stats

export
    # types
    VariateForm,
    ValueSupport,
    Univariate,
    Multivariate,
    Matrixvariate,
    Discrete,
    Continuous,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution,
    NonMatrixDistribution,
    DiscreteDistribution,
    ContinuousDistribution,
    DiscreteUnivariateDistribution,
    DiscreteMultivariateDistribution,
    DiscreteMatrixDistribution,
    ContinuousUnivariateDistribution,
    ContinuousMultivariateDistribution,
    ContinuousMatrixDistribution,
    SufficientStats,
    Arcsine,
    Bernoulli,
    Beta,
    BetaPrime,
    Binomial,
    Categorical,
    Cauchy,
    Chi,
    Chisq,
    Cosine,
    Dirichlet,
    DiscreteUniform,
    DoubleExponential,
    EdgeworthMean,
    EdgeworthSum,
    EdgeworthZ,
    EmpiricalUnivariateDistribution,
    Erlang,
    Exponential,
    FDist,
    Gamma,
    Geometric,
    Gumbel,
    HyperGeometric,
    InverseWishart,
    InverseGamma,
    InverseGaussian,    
    Kolmogorov,
    KSDist,
    KSOneSided,
    Laplace,
    Levy,
    Logistic,
    LogNormal,
    MixtureModel,
    Multinomial,
    MultivariateNormal,
    MvNormal,
    NegativeBinomial,
    NoncentralBeta,
    NoncentralChisq,
    NoncentralF,
    NoncentralT,
    Normal,
    Pareto,
    Poisson,
    Rayleigh,
    Skellam,
    TDist,
    Triangular,
    Truncated,
    Uniform,
    VonMisesFisher,
    Weibull,
    Wishart,
    QQPair,

    # methods
    binaryentropy, # entropy of distribution in bits
    ccdf,          # complementary cdf, i.e. 1 - cdf
    cdf,           # cumulative distribution function
    cf,            # characteristic function
    cgf,           # cumulant generating function
    cquantile,     # complementary quantile (i.e. using prob in right hand tail)
    cumulant,      # cumulants of distribution
    dim,           # sample dimension of multivariate distribution
    entropy,       # entropy of distribution in nats
    fit,           # fit a distribution to data (using default method)
    fit_mle,       # fit a distribution to data using MLE
    fit_mle!,      # fit a distribution to data using MLE (inplace update to initial guess)
    fit_map,       # fit a distribution to data using MAP
    freecumulant,  # free cumulants of distribution
    insupport,     # predicate, is x in the support of the distribution?
    invlogccdf,    # complementary quantile based on log probability
    invlogcdf,     # quantile based on log probability
    isplatykurtic, # Is excess kurtosis > 0.0?
    isleptokurtic, # Is excess kurtosis < 0.0?
    ismesokurtic,  # Is excess kurtosis = 0.0?
    isprobvec,     # Is a probability vector?
    kde,           # Kernel density estimator
    kurtosis,      # kurtosis of the distribution
    logccdf,       # ccdf returning log-probability
    logcdf,        # cdf returning log-probability
    loglikelihood, # log probability of array of IID draws
    logpdf,        # log probability density
    logpdf!,       # evaluate log pdf to provided storage
    logpmf,        # log probability mass
    logpmf!,       # evaluate log pmf to provided storage
    posterior,     # Bayesian updating
    scale,         # scale parameter of a distribution
    rate,          # rate parameter of a distribution
    sqmahal,       # squared Mahalanobis distance to Gaussian center
    sqmahal!,      # inplace evaluation of sqmahal
    mean,          # mean of distribution
    median,        # median of distribution
    mgf,           # moment generating function
    mode,          # the mode of a unimodal distribution
    modes,         # mode(s) of distribution as vector
    moment,        # moments of distribution
    nsamples,      # get the number of samples in a data array based on distribution types
    pdf,           # probability density function (ContinuousDistribution)
    pmf,           # probability mass function (DiscreteDistribution)
    quantile,      # inverse of cdf (defined for p in (0,1))
    qqbuild,       # build a paired quantiles data structure for qqplots
    rand,          # random sampler
    rand!,         # replacement random sampler
    sample,        # sample from a source array
    sampler,       # create a Sampler object for efficient samples
    skewness,      # skewness of the distribution
    sprand,        # random sampler for sparse matrices
    std,           # standard deviation of distribution
    suffstats,     # compute sufficient statistics
    var,           # variance of distribution
    wsample        # weighted sampling from a source array

import Base.mean, Base.median, Base.quantile, Base.max, Base.min, Base.scale
import Base.Random, Base.rand, Base.rand!, Base.std, Base.var, Base.cor, Base.cov
import Base.show, Base.sprand
import NumericExtensions.dim, NumericExtensions.entropy
import Stats.kurtosis, Stats.skewness, Stats.mode, Stats.modes


#### Distribution type system

abstract ValueSupport
type Discrete <: ValueSupport end
type Continuous <: ValueSupport end

abstract VariateForm
type Univariate <: VariateForm end
type Multivariate <: VariateForm end
type Matrixvariate <: VariateForm end

abstract Distribution{F<:VariateForm,S<:ValueSupport}

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

abstract SufficientStats
abstract GenerativeFormulation


#### Include files

include("constants.jl")

include("fallbacks.jl")
include("rmath.jl")
include("specialfuns.jl")
include("tvpack.jl")
include("utils.jl")

include(joinpath("samplers", "categorical_samplers.jl"))

# Univariate distributions
include(joinpath("univariate", "arcsine.jl"))
include(joinpath("univariate", "bernoulli.jl"))
include(joinpath("univariate", "beta.jl"))
include(joinpath("univariate", "betaprime.jl"))
include(joinpath("univariate", "binomial.jl"))
include(joinpath("univariate", "categorical.jl"))
include(joinpath("univariate", "cauchy.jl"))
include(joinpath("univariate", "chi.jl"))
include(joinpath("univariate", "chisq.jl"))
include(joinpath("univariate", "cosine.jl"))
include(joinpath("univariate", "discreteuniform.jl"))
include(joinpath("univariate", "empirical.jl"))
include(joinpath("univariate", "exponential.jl"))
include(joinpath("univariate", "fdist.jl"))
include(joinpath("univariate", "gamma.jl"))
include(joinpath("univariate", "edgeworth.jl"))
include(joinpath("univariate", "erlang.jl"))
include(joinpath("univariate", "geometric.jl"))
include(joinpath("univariate", "gumbel.jl"))
include(joinpath("univariate", "hypergeometric.jl"))
include(joinpath("univariate", "inversegamma.jl"))
include(joinpath("univariate", "inversegaussian.jl"))
include(joinpath("univariate", "kolmogorov.jl"))
include(joinpath("univariate", "ksdist.jl"))
include(joinpath("univariate", "ksonesided.jl"))
include(joinpath("univariate", "laplace.jl"))
include(joinpath("univariate", "levy.jl"))
include(joinpath("univariate", "logistic.jl"))
include(joinpath("univariate", "lognormal.jl"))
include(joinpath("univariate", "negativebinomial.jl"))
include(joinpath("univariate", "noncentralbeta.jl"))
include(joinpath("univariate", "noncentralchisq.jl"))
include(joinpath("univariate", "noncentralf.jl"))
include(joinpath("univariate", "noncentralt.jl"))
include(joinpath("univariate", "normal.jl"))
include(joinpath("univariate", "pareto.jl"))
include(joinpath("univariate", "poisson.jl"))
include(joinpath("univariate", "rayleigh.jl"))
include(joinpath("univariate", "skellam.jl"))
include(joinpath("univariate", "tdist.jl"))
include(joinpath("univariate", "triangular.jl"))
include(joinpath("univariate", "uniform.jl"))
include(joinpath("univariate", "weibull.jl"))

# Multivariate distributions
include(joinpath("multivariate", "dirichlet.jl"))
include(joinpath("multivariate", "multinomial.jl"))
include(joinpath("multivariate", "multivariatenormal.jl"))
include(joinpath("multivariate", "vonmisesfisher.jl"))

# Matrix distributions
include(joinpath("matrix", "inversewishart.jl"))
include(joinpath("matrix", "wishart.jl"))

# Truncated distributions
include("truncate.jl")
include(joinpath("univariate", "truncated", "normal.jl"))

# Mixture distributions
include("mixturemodel.jl")

# Sample w/ and w/o replacement
include("sample.jl")

# Link functions for GLM's
#include("glmtools.jl")

# REPL representations
include("show.jl")

# Kernel density estimators
include("kde.jl")

# Expectations, entropy, KL divergence
include("functionals.jl")

include("conjugates.jl")

include("qq.jl")

include("estimators.jl")

end # module
