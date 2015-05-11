module Distributions

using ArrayViews
using PDMats
using StatsBase
using Compat

import Base.Random
import Base: size, eltype, length, full, convert, show, getindex, scale, rand, rand!
import Base: sum, mean, median, maximum, minimum, quantile, std, var, cov, cor
import Base: +, -, .+, .-
import Base.LinAlg: Cholesky
import Base.Math.@horner
import StatsBase: kurtosis, skewness, entropy, mode, modes, randi, fit, kldivergence
import StatsBase: RandIntSampler
import PDMats: dim, PDMat, invquad

export
    # generic types
    VariateForm,
    ValueSupport,
    Univariate,
    Multivariate,
    Matrixvariate,
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
    Beta,
    BetaPrime,
    Binomial,
    Biweight,
    Categorical,
    Cauchy,
    Chi,
    Chisq,
    Cosine,
    DiagNormal,
    DiagNormalCanon,
    Dirichlet,
    DiscreteUniform,
    DoubleExponential,
    EdgeworthMean,
    EdgeworthSum,
    EdgeworthZ,
    EmpiricalUnivariateDistribution,
    Erlang,
    Epanechnikov,
    Exponential,
    FDist,
    FisherNoncentralHypergeometric,
    Frechet,
    FullNormal,
    FullNormalCanon,
    Gamma,
    Geometric,
    Gumbel,
    Hypergeometric,
    InverseWishart,
    InverseGamma,
    InverseGaussian,
    IsoNormal,
    IsoNormalCanon,
    Kolmogorov,
    KSDist,
    KSOneSided,
    Laplace,
    Levy,
    Logistic,
    LogNormal,
    MaxwellBoltzmann,
    MixtureModel,
    Multinomial,
    MultivariateNormal,
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
    NormalGamma,
    NormalInverseGamma,
    NormalInverseWishart,
    NormalWishart,
    Pareto,
    Poisson,
    PoissonBinomial,
    QQPair,
    Rayleigh,
    Skellam,
    SymTriangularDist,
    TDist,
    TriangularDist,
    Triweight,
    Truncated,
    TruncatedNormal,
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
    binaryentropy,      # entropy of distribution in bits
    canonform,          # get canonical form of a distribution
    ccdf,               # complementary cdf, i.e. 1 - cdf
    cdf,                # cumulative distribution function
    cf,                 # characteristic function
    cgf,                # cumulant generating function
    circmean,           # mean of circular distribution
    circmedian,         # median of circular distribution
    circmode,           # mode of circular distribution
    circvar,            # variance of circular distribution
    cquantile,          # complementary quantile (i.e. using prob in right hand tail)
    cumulant,           # cumulants of distribution
    complete,           # turn an incomplete formulation into a complete distribution
    component,          # get the k-th component of a mixture model
    components,         # get components from a mixture model
    componentwise_pdf,      # component-wise pdf for mixture models
    componentwise_logpdf,   # component-wise logpdf for mixture models
    concentration,      # the concentration parameter
    dim,                # sample dimension of multivariate distribution
    dof,                # get the degree of freedom
    entropy,            # entropy of distribution in nats
    failprob,           # failing probability
    fit,                # fit a distribution to data (using default method)
    fit_mle,            # fit a distribution to data using MLE
    fit_mle!,           # fit a distribution to data using MLE (inplace update to initial guess)
    fit_map,            # fit a distribution to data using MAP
    freecumulant,       # free cumulants of distribution
    gmvnormal,          # a generic function to construct multivariate normal distributions
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
    kde,                # Kernel density estimator (from Stats.jl)
    kurtosis,           # kurtosis of the distribution
    logccdf,            # ccdf returning log-probability
    logcdf,             # cdf returning log-probability
    logdetcov,          # log-determinant of covariance
    loglikelihood,      # log probability of array of IID draws
    logpdf,             # log probability density
    logpdf!,            # evaluate log pdf to provided storage
    logpmf,             # log probability mass
    logpmf!,            # evaluate log pmf to provided storage
    posterior,          # get posterior distribution given prior and observed data
    posterior_canon,    # get the canonical form of the posterior distribution
    posterior_mode,     # get the mode of posterior distribution
    posterior_rand,     # draw samples from the posterior distribution
    posterior_rand!,
    posterior_randmodel,

    invscale,           # Inverse scale parameter
    sqmahal,            # squared Mahalanobis distance to Gaussian center
    sqmahal!,           # inplace evaluation of sqmahal
    location,           # get the location parameter
    mean,               # mean of distribution
    meandir,            # mean direction (of a spherical distribution)
    meanform,           # convert a normal distribution from canonical form to mean form
    meanlogx,           # the mean of log(x)
    median,             # median of distribution
    mgf,                # moment generating function
    mode,               # the mode of a unimodal distribution
    modes,              # mode(s) of distribution as vector
    moment,             # moments of distribution
    nsamples,           # get the number of samples contained in an array
    ncategories,        # the number of categories in a Categorical distribution
    ncomponents,        # the number of components in a mixture model
    ntrials,            # the number of trials being performed in the experiment
    params,             # get the tuple of parameters
    pdf,                # probability density function (ContinuousDistribution)
    pmf,                # probability mass function (DiscreteDistribution)
    probs,              # Get the vector of probabilities
    probval,            # The pdf/pmf value for a uniform distribution
    quantile,           # inverse of cdf (defined for p in (0,1))
    qqbuild,            # build a paired quantiles data structure for qqplots
    rate,               # get the rate parameter
    sampler,            # create a Sampler object for efficient samples
    scale,              # get the scale parameter
    shape,              # get the shape parameter
    skewness,           # skewness of the distribution
    span,               # the span of the support, e.g. maximum(d) - minimum(d)
    std,                # standard deviation of distribution
    stdlogx,            # standard deviation of log(x)
    suffstats,          # compute sufficient statistics
    succprob,           # the success probability
    support,            # the support of a distribution (or a distribution type)
    test_samples,       # test a sampler
    test_distr,         # test a distribution
    var,                # variance of distribution
    varlogx,            # variance of log(x)
    expected_logdet,    # expected logarithm of random matrix determinant
    gradlogpdf,         # gradient (or derivative) of logpdf(d,x) wrt x

    # reexport from StatsBase
    sample, sample!,        # sample from a source array
    wsample, wsample!      # weighted sampling from a source array


### source files

# type system
include("common.jl")

# implementation helpers
include("constants.jl")
include("specialfuns.jl")
include("tvpack.jl")
include("utils.jl")
include("rmath.jl")

# generic functions
include("show.jl")
include("genericrand.jl")
include("functionals.jl")
include("genericfit.jl")

# specific samplers and distributions
include("univariates.jl")
include("empirical.jl")
include("edgeworth.jl")
include("multivariates.jl")
include("matrixvariates.jl")
include("samplers.jl")

# others
include("truncate.jl")
include("qq.jl")
include("estimators.jl")
include("testutils.jl")

# mixture distributions (TODO: moveout)
include("mixtures/mixturemodel.jl")
include("mixtures/unigmm.jl")

include("deprecates.jl")

end # module
