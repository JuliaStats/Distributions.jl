module Distributions

using StatsBase, PDMats, StatsFuns, Statistics
using StatsFuns: logtwo, invsqrt2, invsqrt2π

import QuadGK: quadgk
import Base: size, length, convert, show, getindex, rand, vec, inv
import Base: sum, maximum, minimum, extrema, +, -, *, ==
import Base.Math: @horner

using FillArrays

using LinearAlgebra, Printf
import LinearAlgebra: dot, rank

using Random
import Random: default_rng, rand!, SamplerRangeInt

import Statistics: mean, median, quantile, std, var, cov, cor
import StatsAPI
import StatsBase: kurtosis, skewness, entropy, mode, modes,
                  fit, kldivergence, loglikelihood, dof, span,
                  params, params!

import PDMats: dim, PDMat, invquad

using SpecialFunctions
using Base.MathConstants: eulergamma

import AliasTables

export
    # re-export Statistics
    mean, median, quantile, std, var, cov, cor,

    # generic types
    VariateForm,
    ArrayLikeVariate,
    ValueSupport,
    Univariate,
    Multivariate,
    Matrixvariate,
    CholeskyVariate,
    NamedTupleVariate,
    Discrete,
    Continuous,
    Sampleable,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution,
    NoncentralHypergeometric,
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
    AbstractMvNormal,
    AbstractMixtureModel,
    UnivariateMixture,
    MultivariateMixture,

    # distribution types
    Arcsine,
    Bernoulli,
    BernoulliLogit,
    Beta,
    BetaBinomial,
    BetaPrime,
    Binomial,
    Biweight,
    Categorical,
    Cauchy,
    Chernoff,
    Chi,
    Chisq,
    Cosine,
    DiagNormal,
    DiagNormalCanon,
    Dirac,
    Dirichlet,
    DirichletMultinomial,
    DiscreteUniform,
    DoubleExponential,
    EdgeworthMean,
    EdgeworthSum,
    EdgeworthZ,
    Erlang,
    Epanechnikov,
    Exponential,
    FDist,
    FisherNoncentralHypergeometric,
    Frechet,
    FullNormal,
    FullNormalCanon,
    Gamma,
    DiscreteNonParametric,
    GeneralizedPareto,
    GeneralizedExtremeValue,
    Geometric,
    Gumbel,
    Hypergeometric,
    InverseWishart,
    InverseGamma,
    InverseGaussian,
    IsoNormal,
    IsoNormalCanon,
    JohnsonSU,
    JointOrderStatistics,
    Kolmogorov,
    KSDist,
    KSOneSided,
    Kumaraswamy,
    Laplace,
    Levy,
    Lindley,
    LKJ,
    LKJCholesky,
    LocationScale,
    Logistic,
    LogNormal,
    LogUniform,
    MvLogitNormal,
    LogitNormal,
    MatrixBeta,
    MatrixFDist,
    MatrixNormal,
    MatrixTDist,
    MixtureModel,
    Multinomial,
    MultivariateNormal,
    MvLogNormal,
    MvNormal,
    MvNormalCanon,
    MvNormalKnownCov,
    MvTDist,
    NegativeBinomial,
    NoncentralBeta,
    NoncentralChisq,
    NoncentralF,
    NoncentralHypergeometric,
    NoncentralT,
    Normal,
    NormalCanon,
    NormalInverseGaussian,
    OrderStatistic,
    Pareto,
    PGeneralizedGaussian,
    SkewedExponentialPower,
    Product, # deprecated
    Poisson,
    PoissonBinomial,
    QQPair,
    Rayleigh,
    Rician,
    Semicircle,
    Skellam,
    SkewNormal,
    Soliton,
    StudentizedRange,
    SymTriangularDist,
    TDist,
    TriangularDist,
    Triweight,
    Truncated,
    Uniform,
    UnivariateGMM,
    VonMises,
    VonMisesFisher,
    WalleniusNoncentralHypergeometric,
    Weibull,
    Wishart,
    ZeroMeanIsoNormal,
    ZeroMeanIsoNormalCanon,
    ZeroMeanDiagNormal,
    ZeroMeanDiagNormalCanon,
    ZeroMeanFullNormal,
    ZeroMeanFullNormalCanon,

    # auxiliary types
    RealInterval,

    # methods
    canonform,          # get canonical form of a distribution
    ccdf,               # complementary cdf, i.e. 1 - cdf
    cdf,                # cumulative distribution function
    censored,           # censor a distribution with a lower and upper bound
    cf,                 # characteristic function
    cquantile,          # complementary quantile (i.e. using prob in right hand tail)
    component,          # get the k-th component of a mixture model
    components,         # get components from a mixture model
    componentwise_pdf,      # component-wise pdf for mixture models
    componentwise_logpdf,   # component-wise logpdf for mixture models
    concentration,      # the concentration parameter
    convolve,           # convolve distributions of the same type
    dof,                # get the degree of freedom
    entropy,            # entropy of distribution in nats
    failprob,           # failing probability
    fit,                # fit a distribution to data (using default method)
    fit_mle,            # fit a distribution to data using MLE
    insupport,          # predicate, is x in the support of the distribution?
    invcov,             # get the inversed covariance
    invlogccdf,         # complementary quantile based on log probability
    invlogcdf,          # quantile based on log probability
    isplatykurtic,      # Is excess kurtosis > 0.0?
    isleptokurtic,      # Is excess kurtosis < 0.0?
    ismesokurtic,       # Is excess kurtosis = 0.0?
    isprobvec,          # Is a probability vector?
    isupperbounded,
    islowerbounded,
    isbounded,
    hasfinitesupport,
    kldivergence,       # kl divergence between distributions
    kurtosis,           # kurtosis of the distribution
    logccdf,            # ccdf returning log-probability
    logcdf,             # cdf returning log-probability
    logdiffcdf,         # log of difference between cdf at two values
    logdetcov,          # log-determinant of covariance
    loglikelihood,      # log probability of array of IID draws
    logpdf,             # log probability density
    logpdf!,            # evaluate log pdf to provided storage

    invscale,           # Inverse scale parameter
    sqmahal,            # squared Mahalanobis distance to Gaussian center
    sqmahal!,           # in-place evaluation of sqmahal
    location,           # get the location parameter
    location!,          # provide storage for the location parameter (used in multivariate distribution mvlognormal)
    mean,               # mean of distribution
    meandir,            # mean direction (of a spherical distribution)
    meanform,           # convert a normal distribution from canonical form to mean form
    meanlogx,           # the mean of log(x)
    median,             # median of distribution
    mgf,                # moment generating function
    cgf,                # cumulant generating function
    mode,               # the mode of a unimodal distribution
    modes,              # mode(s) of distribution as vector
    moment,             # moments of distribution
    nsamples,           # get the number of samples contained in an array
    ncategories,        # the number of categories in a Categorical distribution
    ncomponents,        # the number of components in a mixture model
    ntrials,            # the number of trials being performed in the experiment
    params,             # get the tuple of parameters
    params!,            # provide storage space to calculate the tuple of parameters for a multivariate distribution like mvlognormal
    partype,            # returns a type large enough to hold all of a distribution's parameters' element types
    pdf,                # probability density function (ContinuousDistribution)
    pdfsquaredL2norm,   # squared L2 norm of the probability density function
    probs,              # Get the vector of probabilities
    probval,            # The pdf/pmf value for a uniform distribution
    product_distribution, # product of univariate distributions
    quantile,           # inverse of cdf (defined for p in (0,1))
    qqbuild,            # build a paired quantiles data structure for qqplots
    rate,               # get the rate parameter
    sampler,            # create a Sampler object for efficient samples
    scale,              # get the scale parameter
    scale!,             # provide storage for the scale parameter (used in multivariate distribution mvlognormal)
    shape,              # get the shape parameter
    skewness,           # skewness of the distribution
    span,               # the span of the support, e.g. maximum(d) - minimum(d)
    std,                # standard deviation of distribution
    stdlogx,            # standard deviation of log(x)
    suffstats,          # compute sufficient statistics
    succprob,           # the success probability
    support,            # the support of a distribution (or a distribution type)
    truncated,          # truncate a distribution with a lower and upper bound
    var,                # variance of distribution
    varlogx,            # variance of log(x)
    expected_logdet,    # expected logarithm of random matrix determinant
    gradlogpdf,         # gradient (or derivative) of logpdf(d,x) wrt x

    # reexport from StatsBase
    sample, sample!,        # sample from a source array
    wsample, wsample!       # weighted sampling from a source array

### source files

# type system
include("common.jl")

# implementation helpers
include("utils.jl")
include("eachvariate.jl")

# generic functions
include("show.jl")
include("quantilealgs.jl")
include("genericrand.jl")
include("functionals.jl")
include("genericfit.jl")

# specific samplers and distributions
include("univariates.jl")
include("edgeworth.jl")
include("multivariates.jl")
include("matrixvariates.jl")
include("namedtuple/productnamedtuple.jl")
include("cholesky/lkjcholesky.jl")
include("samplers.jl")

# others
include("product.jl")
include("reshaped.jl")
include("truncate.jl")
include("censored.jl")
include("conversion.jl")
include("convolution.jl")
include("qq.jl")
include("estimators.jl")
include("pdfnorm.jl")

# mixture distributions (TODO: moveout)
include("mixtures/mixturemodel.jl")
include("mixtures/unigmm.jl")

# Interface for StatsAPI
include("statsapi.jl")

# Testing utilities for other packages which implement distributions.
include("test_utils.jl")

# Extensions: Implementation of DensityInterface and ChainRulesCore API
if !isdefined(Base, :get_extension)
    include("../ext/DistributionsChainRulesCoreExt/DistributionsChainRulesCoreExt.jl")
    include("../ext/DistributionsDensityInterfaceExt.jl")
    include("../ext/DistributionsTestExt.jl")
end

include("deprecates.jl")

"""
A Julia package for probability distributions and associated functions.

API overview (major features):

- `d = Dist(parameters...)` creates a distribution instance `d` for some distribution `Dist` (see choices below) with the specified `parameters`
- `rand(d, sz)` samples from the distribution
- `pdf(d, x)` and `logpdf(d, x)` compute the probability density or log-probability density of `d` at `x`
- `cdf(d, x)` and `ccdf(d, x)` compute the (complementary) cumulative distribution function at `x`
- `quantile(d, p)` is the inverse `cdf` (see also `cquantile`)
- `mean(d)`, `var(d)`, `std(d)`, `skewness(d)`, `kurtosis(d)` compute moments of `d`
- `fit(Dist, xs)` generates a distribution of type `Dist` that best fits the samples in `xs`

These represent just a few of the operations supported by this
package; users are encouraged to refer to the full documentation at
https://JuliaStats.github.io/Distributions.jl/stable/ for further
information.

Supported distributions:

    Arcsine, Bernoulli, Beta, BetaBinomial, BetaPrime, Binomial, Biweight,
    Categorical, Cauchy, Censored, Chi, Chisq, Cosine, DiagNormal, DiagNormalCanon,
    Dirichlet, DiscreteUniform, DoubleExponential, EdgeworthMean,
    EdgeworthSum, EdgeworthZ, Erlang,
    Epanechnikov, Exponential, FDist, FisherNoncentralHypergeometric,
    Frechet, FullNormal, FullNormalCanon, Gamma, GeneralizedPareto,
    GeneralizedExtremeValue, Geometric, Gumbel, Hypergeometric,
    InverseWishart, InverseGamma, InverseGaussian, IsoNormal,
    IsoNormalCanon, JohnsonSU, Kolmogorov, KSDist, KSOneSided, Kumaraswamy,
    Laplace, Levy, Lindley, LKJ, LKJCholesky,
    Logistic, LogNormal, MatrixBeta, MatrixFDist, MatrixNormal,
    MatrixTDist, MixtureModel, Multinomial,
    MultivariateNormal, MvLogNormal, MvNormal, MvNormalCanon,
    MvNormalKnownCov, MvTDist, NegativeBinomial, NoncentralBeta, NoncentralChisq,
    NoncentralF, NoncentralHypergeometric, NoncentralT, Normal, NormalCanon,
    NormalInverseGaussian, Pareto, PGeneralizedGaussian, Poisson, PoissonBinomial,
    QQPair, Rayleigh, Rician, Skellam, Soliton, StudentizedRange, SymTriangularDist, TDist, TriangularDist,
    Triweight, Truncated, Uniform, UnivariateGMM,
    VonMises, VonMisesFisher, WalleniusNoncentralHypergeometric, Weibull,
    Wishart, ZeroMeanIsoNormal, ZeroMeanIsoNormalCanon,
    ZeroMeanDiagNormal, ZeroMeanDiagNormalCanon, ZeroMeanFullNormal,
    ZeroMeanFullNormalCanon

"""
Distributions

end # module
