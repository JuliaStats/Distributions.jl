module Distributions

using NumericExtensions
using Stats

export                                  # types
#    CauchitLink,
#    CloglogLink,
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
    EmpiricalUnivariateDistribution,
    Erlang,
    Exponential,
    FDist,
    Gamma,
    Geometric,
    Gumbel,
    HyperGeometric,
#    IdentityLink,
#    InverseLink,
    InverseWishart,
    InvertedGamma,
    Laplace,
    Levy,
#    Link,
    Logistic,
#    LogitLink,
#    LogLink,
    LogNormal,
    MixtureModel,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    NoncentralBeta,
    NoncentralChisq,
    NoncentralF,
    NoncentralT,
    Normal,
    Pareto,
    Poisson,
#    ProbitLink,
    Rayleigh,
    Skellam,
    TDist,
    Triangular,
    TruncatedNormal,
    TruncatedUnivariateDistribution,
    Uniform,
    Weibull,
    Wishart,
                                        # methods
    binaryentropy, # entropy of distribution in bits
#    canonicallink, # canonical link function for a distribution
    ccdf,          # complementary cdf, i.e. 1 - cdf
    cdf,           # cumulative distribution function
    cf,            # characteristic function
    cgf,           # cumulant generating function
    cquantile,     # complementary quantile (i.e. using prob in right hand tail)
    cumulant,      # cumulants of distribution
#    deviance,      # deviance of fitted and observed responses
#    devresid,      # vector of squared deviance residuals
    dim,           # sample dimension of multivariate distribution
    entropy,       # entropy of distribution in nats
    fit,           # fit a distribution to data (using default method)
    fit_mle,       # fit a distribution to data using MLE
    fit_map,       # fit a distribution to data using MAP
    freecumulant,  # free cumulants of distribution
    insupport,     # predicate, is x in the support of the distribution?
    invlogccdf,    # complementary quantile based on log probability
    invlogcdf,     # quantile based on log probability
    isplatykurtic, # Is excess kurtosis > 0.0?
    isleptokurtic, # Is excess kurtosis < 0.0?
    ismesokurtic,  # Is excess kurtosis = 0.0?
    kde,           # Kernel density estimator
    kurtosis,      # kurtosis of the distribution
#    linkfun,       # link function mapping mu to eta, the linear predictor
#    linkinv,       # inverse link mapping eta to mu
    logccdf,       # ccdf returning log-probability
    logcdf,        # cdf returning log-probability
    loglikelihood, # log probability of array of IID draws
    logpdf,        # log probability density
    logpdf!,       # evaluate log pdf to provided storage
    logpmf,        # log probability mass
    logpmf!,       # evaluate log pmf to provided storage
    mean,          # mean of distribution
    median,        # median of distribution
    mgf,           # moment generating function
    modes,         # mode(s) of distribution as vector
    moment,        # moments of distribution
#    mueta,         # derivative of inverse link function
#    mustart,       # starting values of mean vector in GLMs
    pdf,           # probability density function (ContinuousDistribution)
    pmf,           # probability mass function (DiscreteDistribution)
    quantile,      # inverse of cdf (defined for p in (0,1))
    rand,          # random sampler
    rand!,         # replacement random sampler
    sample,        # another random sampler - not sure why this is here
    skewness,      # skewness of the distribution
    sprand,        # random sampler for sparse matrices
    std,           # standard deviation of distribution
    valideta,      # validity check on linear predictor
    validmu,       # validity check on mean vector
    var            # variance of distribution

import Base.mean, Base.median, Base.quantile
import Base.rand, Base.std, Base.var, Base.cor, Base.cov
import Base.show, Base.sprand
import Stats.kurtosis, Stats.skewness, Stats.modes

include("drawtable.jl")
include("huffman.jl")
include("alias.jl")
include("tvpack.jl")

abstract Distribution
abstract UnivariateDistribution             <: Distribution
abstract MultivariateDistribution           <: Distribution
abstract MatrixDistribution                 <: Distribution

abstract DiscreteUnivariateDistribution     <: UnivariateDistribution
abstract ContinuousUnivariateDistribution   <: UnivariateDistribution

abstract DiscreteMultivariateDistribution   <: MultivariateDistribution
abstract ContinuousMultivariateDistribution <: MultivariateDistribution

abstract ContinuousMatrixDistribution       <: MatrixDistribution
abstract DiscreteMatrixDistribution         <: MatrixDistribution

typealias NonMatrixDistribution Union(UnivariateDistribution, MultivariateDistribution)
typealias DiscreteDistribution Union(DiscreteUnivariateDistribution, DiscreteMultivariateDistribution)
typealias ContinuousDistribution Union(ContinuousUnivariateDistribution, ContinuousMultivariateDistribution)

include("fallbacks.jl")
include("rmath.jl")

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
include(joinpath("univariate", "erlang.jl"))
include(joinpath("univariate", "geometric.jl"))
include(joinpath("univariate", "gumbel.jl"))
include(joinpath("univariate", "hypergeometric.jl"))
include(joinpath("univariate", "invertedgamma.jl"))
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

end # module
