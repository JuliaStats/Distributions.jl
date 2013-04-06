module Distributions

export                                  # types
    CauchitLink,
    CloglogLink,
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
    Chisq,
    Dirichlet,
    DiscreteUniform,
    EmpiricalDistribution,
    Erlang,
    Exponential,
    FDist,
    Gamma,
    Geometric,
    HyperGeometric,
    IdentityLink,
    InverseLink,
    InverseWishart,
    InvertedGamma,
    Laplace,
    Levy,
    Link,
    Logistic,
    LogitLink,
    LogLink,
    logNormal,
    MixtureModel,
    MStDist,
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
    ProbitLink,
    Rayleigh,
    StDist,
    TDist,
    Triangular,
    Uniform,
    Weibull,
    Wishart,
                                        # methods
    binaryentropy, # entropy of distribution in bits
    canonicallink, # canonical link function for a distribution
    ccdf,          # complementary cdf, i.e. 1 - cdf
    cdf,           # cumulative distribution function
    cf,            # characteristic function
    cgf,           # cumulant generating function
    cquantile,     # complementary quantile (i.e. using prob in right hand tail)
    cumulant,      # cumulants of distribution
    deviance,      # deviance of fitted and observed responses
    devresid,      # vector of squared deviance residuals
    entropy,       # entropy of distribution in nats
    fit,           # fit a distribution to data
    freecumulant,  # free cumulants of distribution
    insupport,     # predicate, is x in the support of the distribution?
    invlogccdf,    # complementary quantile based on log probability
    invlogcdf,     # quantile based on log probability
    kurtosis,      # kurtosis of the distribution
    linkfun,       # link function mapping mu to eta, the linear predictor
    linkinv,       # inverse link mapping eta to mu
    logccdf,       # ccdf returning log-probability
    logcdf,        # cdf returning log-probability
    logpdf,        # log probability density
    logpmf,        # log probability mass
    mean,          # mean of distribution
    median,        # median of distribution
    mgf,           # moment generating function
    modes,         # mode(s) of distribution as vector
    moment,        # moments of distribution
    mueta,         # derivative of inverse link function
    mustart,       # starting values of mean vector in GLMs
    pdf,           # probability density function (ContinuousDistribution)
    pmf,           # probability mass function (DiscreteDistribution)
    quantile,      # inverse of cdf (defined for p in (0,1))
    rand,          # random sampler
    rand!,         # replacement random sampler
    sample,        # another random sampler - not sure why this is here
    skewness,      # skewness of the distribution
    std,           # standard deviation of distribution
    valideta,      # validity check on linear predictor
    validmu,       # validity check on mean vector
    var            # variance of distribution

import Base.mean, Base.median, Base.quantile
import Base.rand, Base.std, Base.var, Base.integer_valued
import Base.show

# convenience methods for integer_valued
integer_valued{T <: Integer}(x::AbstractArray{T}) = true
integer_valued(x::AbstractArray) = allp(integer_valued, x)

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
abstract DiscreteMatrixDistribution         <: MatrixDistribution # is there such a thing?

typealias NonMatrixDistribution Union(UnivariateDistribution, MultivariateDistribution)
typealias DiscreteDistribution Union(DiscreteUnivariateDistribution, DiscreteMultivariateDistribution)
typealias ContinuousDistribution Union(ContinuousUnivariateDistribution, ContinuousMultivariateDistribution)

## Fallback methods, usually overridden for specific distributions
ccdf(d::Distribution, q::Real)                = 1.0 - cdf(d, q)
cquantile(d::Distribution, p::Real)           = quantile(d, 1.0 - p)
function deviance{M<:Real,Y<:Real,W<:Real}(d::Distribution,
                                           mu::AbstractArray{M},
                                           y::AbstractArray{Y},
                                           wt::AbstractArray{W})
    promote_shape(size(mu), promote_shape(size(y), size(wt)))
    ans = 0.
    for i in 1:length(y)
        ans += wt[i] * logpdf(d, mu[i], y[i])
    end
    -2ans
end
devresid(d::Distribution, y::Real, mu::Real, wt::Real) = -2wt*logpdf(d, mu, y)
function devresid{Y<:Real,M<:Real,W<:Real}(d::Distribution,
                                           y::AbstractArray{Y},
                                           mu::AbstractArray{M},
                                           wt::AbstractArray{W})
    R = Array(Float64, promote_shape(size(y), promote_shape(size(mu), size(wt))))
    for i in 1:length(mu)
        R[i] = devresid(d, y[i], mu[i], wt[i])
    end
    R
end
function devresid(d::Distribution, y::Vector{Float64}, mu::Vector{Float64}, wt::Vector{Float64})
    [devresid(d, y[i], mu[i], wt[i]) for i in 1:length(y)]
end
invlogccdf(d::Distribution, lp::Real)         = quantile(d, exp(-lp))
invlogcdf(d::Distribution, lp::Real)          = quantile(d, exp(lp))
logccdf(d::Distribution, q::Real)             = log(ccdf(d,q))
logcdf(d::Distribution, q::Real)              = log(cdf(d,q))
logpdf(d::Distribution, x::Real)    = log(pdf(d,x))
function mustart{Y<:Real,W<:Real}(d::Distribution,
                                  y::AbstractArray{Y},
                                  wt::AbstractArray{W})
    M = Array(Float64, promote_shape(size(y), size(wt)))
    for i in 1:length(M)
        M[i] = mustart(d, y[i], wt[i])
    end
    M
end
std(d::Distribution)                          = sqrt(var(d))

#kurtosis returns excess kurtosis by default. proper kurtosis can be returned with correction=false
kurtosis(d::Distribution, correction::Bool)   = correction ? kurtosis(d) : kurtosis(d) + 3.0
excess(d) = kurtosis(d)
excess_kurtosis(d) = kurtosis(d)
proper_kurtosis(d) = kurtosis(d, false)

function rand!(d::UnivariateDistribution, A::Array)
    for i in 1:length(A) A[i] = rand(d) end
    A
end
rand(d::ContinuousDistribution, dims::Dims)   = rand!(d, Array(Float64, dims))
rand(d::DiscreteDistribution, dims::Dims)     = rand!(d, Array(Int,dims))
rand(d::NonMatrixDistribution, dims::Int...) = rand(d, dims)
rand(d::MultivariateDistribution, dims::Int)  = rand(d, (dims, length(mean(d))))
rand(d::MatrixDistribution, dims::Int) = rand!(d, Array(Matrix{Float64},dims))

function rand!(d::MultivariateDistribution, X::Matrix)
  k = length(mean(d))
  m, n = size(X)
  if m == k
    for i = 1:n
      X[:,i] = rand(d)
    end
  elseif n == k
    for i = 1:m
      X[i,:] = rand(d)
    end
  else
    error("Wrong dimensions")
  end
  return X
end

function rand!(d::MatrixDistribution, X::Array{Matrix{Float64}})
  for i in 1:length(X)
    X[i] = rand(d)
  end
  return X
end

function var{M<:Real}(d::Distribution, mu::AbstractArray{M})
    V = similar(mu, Float64)
    for i in 1:length(mu)
        V[i] = var(d, mu[i])
    end
    V
end

function insupport{T<:Real}(d::Distribution, x::AbstractArray{T})
    for e in x
        if !insupport(d, e)
            return false
        end
    end
    true
end
const Rmath = :libRmath
## FIXME: Replace the three _jl_dist_*p macros with one by defining
## the argument tuples for the ccall dynamically from pn
macro _jl_dist_1p(T, b)
    dd = Expr(:quote, string("d", b))     # C name for pdf
    pp = Expr(:quote, string("p", b))     # C name for cdf
    qq = Expr(:quote, string("q", b))     # C name for quantile
    rr = Expr(:quote, string("r", b))     # C name for random sampler
    Ty = eval(T)
    dc = Ty <: DiscreteDistribution
    pn = Ty.names                       # parameter names
    p  = Expr(:quote, pn[1])
    quote
        global pdf, logpdf, cdf, logcdf, ccdf, logccdf
        global quantile, cquantile, invlogcdf, invlogccdf, rand
        function pdf(d::($T), x::Real)
            ccall(($dd, Rmath), Float64,
                  (Float64, Float64, Int32),
                  x, d.($p), 0)
        end
        function logpdf(d::($T), x::Real)
            ccall(($dd, Rmath), Float64,
                  (Float64, Float64, Int32),
                  x, d.($p), 1)
        end
        function cdf(d::($T), q::Real)
            ccall(($pp, Rmath), Float64,
                  (Float64, Float64, Int32, Int32),
                  q, d.($p), 1, 0)
        end
        function logcdf(d::($T), q::Real)
            ccall(($pp, Rmath), Float64,
                  (Float64, Float64, Int32, Int32),
                  q, d.($p), 1, 1)
        end
        function ccdf(d::($T), q::Real)
            ccall(($pp, Rmath),
                  Float64, (Float64, Float64, Int32, Int32),
                  q, d.($p), 0, 0)
        end
        function logccdf(d::($T), q::Real)
            ccall(($pp, Rmath), Float64, (Float64, Float64, Int32, Int32),
                  q, d.($p), 0, 1)
        end
        function quantile(d::($T), p::Real)
            ccall(($qq, Rmath), Float64, (Float64, Float64, Int32, Int32),
                  p, d.($p), 1, 0)
        end
        function cquantile(d::($T), p::Real)
            ccall(($qq, Rmath), Float64, (Float64, Float64, Int32, Int32),
                  p, d.($p), 0, 0)
        end
        function invlogcdf(d::($T), lp::Real)
            ccall(($qq, Rmath), Float64, (Float64, Float64, Int32, Int32),
                  lp, d.($p), 1, 1)
        end
        function invlogccdf(d::($T), lp::Real)
            ccall(($qq, Rmath), Float64, (Float64, Float64, Int32, Int32),
                  lp, d.($p), 0, 1)
        end
        if $dc
            function rand(d::($T))
                int(ccall(($rr, Rmath), Float64, (Float64,), d.($p)))
            end
        else
            function rand(d::($T))
                ccall(($rr, Rmath), Float64, (Float64,), d.($p))
            end
        end
    end
end

macro _jl_dist_2p(T, b)
    dd = Expr(:quote, string("d",b))     # C name for pdf
    pp = Expr(:quote, string("p",b))     # C name for cdf
    qq = Expr(:quote, string("q",b))     # C name for quantile
    rr = Expr(:quote, string("r",b))     # C name for random sampler
    Ty = eval(T)
    dc = Ty <: DiscreteDistribution
    pn = Ty.names                       # parameter names
    p1 = Expr(:quote, pn[1])
    p2 = Expr(:quote, pn[2])
    if string(b) == "norm"              # normal dist has unusual names
        dd = Expr(:quote, :dnorm4)
        pp = Expr(:quote, :pnorm5)
        qq = Expr(:quote, :qnorm5)
    end
    quote
        global pdf, logpdf, cdf, logcdf, ccdf, logccdf
        global quantile, cquantile, invlogcdf, invlogccdf, rand
        function pdf(d::($T), x::Real)
            ccall(($dd, Rmath),
                  Float64, (Float64, Float64, Float64, Int32),
                  x, d.($p1), d.($p2), 0)
        end
        function logpdf(d::($T), x::Real)
            ccall(($dd, Rmath),
                  Float64, (Float64, Float64, Float64, Int32),
                  x, d.($p1), d.($p2), 1)
        end
        function cdf(d::($T), q::Real)
            ccall(($pp, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), 1, 0)
        end
        function logcdf(d::($T), q::Real)
            ccall(($pp, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), 1, 1)
        end
        function ccdf(d::($T), q::Real)
            ccall(($pp, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), 0, 0)
        end
        function logccdf(d::($T), q::Real)
            ccall(($pp, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), 0, 1)
        end
        function quantile(d::($T), p::Real)
            ccall(($qq, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  p, d.($p1), d.($p2), 1, 0)
        end
        function cquantile(d::($T), p::Real)
            ccall(($qq, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  p, d.($p1), d.($p2), 0, 0)
        end
        function invlogcdf(d::($T), lp::Real)
            ccall(($qq, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  lp, d.($p1), d.($p2), 1, 1)
        end
        function invlogccdf(d::($T), lp::Real)
            ccall(($qq, Rmath),
                  Float64, (Float64, Float64, Float64, Int32, Int32),
                  lp, d.($p1), d.($p2), 0, 1)
        end
        if $dc
            function rand(d::($T))
                int(ccall(($rr, Rmath), Float64,
                          (Float64,Float64), d.($p1), d.($p2)))
            end
        else
            function rand(d::($T))
                ccall(($rr, Rmath), Float64,
                      (Float64,Float64), d.($p1), d.($p2))
            end
        end
    end
end

macro _jl_dist_3p(T, b)
    dd = Expr(:quote, string("d", b))     # C name for pdf
    pp = Expr(:quote, string("p", b))     # C name for cdf
    qq = Expr(:quote, string("q", b))     # C name for quantile
    rr = Expr(:quote, string("r", b))     # C name for random sampler
    Ty = eval(T)
    dc = Ty <: DiscreteDistribution
    pn = Ty.names                       # parameter names
    p1 = Expr(:quote, pn[1])
    p2 = Expr(:quote, pn[2])
    p3 = Expr(:quote, pn[3])
    quote
        global pdf, logpdf, cdf, logcdf, ccdf, logccdf
        global quantile, cquantile, invlogcdf, invlogccdf, rand
        function pdf(d::($T), x::Real)
            ccall(($dd, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32),
                  x, d.($p1), d.($p2), d.($p3), 0)
        end
        function logpdf(d::($T), x::Real)
            ccall(($dd, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32),
                  x, d.($p1), d.($p2), d.($p3), 1)
        end
        function cdf(d::($T), q::Real)
            ccall(($pp, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), d.($p3), 1, 0)
        end
        function logcdf(d::($T), q::Real)
            ccall(($pp, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), d.($p3), 1, 1)
        end
        function ccdf(d::($T), q::Real)
            ccall(($pp, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), d.($p3), 0, 0)
        end
        function logccdf(d::($T), q::Real)
            ccall(($pp, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  q, d.($p1), d.($p2), d.($p3), 0, 1)
        end
        function quantile(d::($T), p::Real)
            ccall(($qq, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  p, d.($p1), d.($p2), d.($p3), 1, 0)
        end
        function cquantile(d::($T), p::Real)
            ccall(($qq, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  p, d.($p1), d.($p2), d.($p3), 0, 0)
        end
        function invlogcdf(d::($T), lp::Real)
            ccall(($qq, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  lp, d.($p1), d.($p2), d.($p3), 1, 1)
        end
        function invlogccdf(d::($T), lp::Real)
            ccall(($qq, Rmath), Float64,
                  (Float64, Float64, Float64, Float64, Int32, Int32),
                  lp, d.($p1), d.($p2), d.($p3), 0, 1)
        end
        if $dc
            function rand(d::($T))
                int(ccall(($rr, Rmath), Float64,
                          (Float64,Float64,Float64), d.($p1), d.($p2), d.($p3)))
            end
        else
            function rand(d::($T))
                ccall(($rr, Rmath), Float64,
                      (Float64,Float64,Float64), d.($p1), d.($p2), d.($p3))
            end
        end
    end
end

for f in (:pdf, :logpdf, :cdf, :logcdf,
          :ccdf, :logccdf, :quantile, :cquantile,
          :invlogcdf, :invlogccdf)
  @eval begin
    function ($f)(d::UnivariateDistribution, x::AbstractArray)
      res = Array(Float64, size(x))
      for i in 1:length(res)
        res[i] = ($f)(d, x[i])
      end
      return res
    end
  end
end

pmf(d::DiscreteDistribution, args::Any...) = pdf(d, args...)
logpmf(d::DiscreteDistribution, args::Any...) = logpdf(d, args...)

binary_entropy(d::Distribution) = entropy(d) / log(2)

##############################################################################
#
# Alpha distribution
#
# TODO: Decide what to implement here. Alpha-stable distributions are often
#  analytically complex, so it may be not possible to define useful samplers.
#
##############################################################################

immutable Alpha <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Alpha(l::Real, s::Real)
      s < 0. ? error("scale must be non-negative") : new(float64(l), float64(s))
    end
end
Alpha(location::Real) = Alpha(location, 1.0)
Alpha() = Alpha(0.0, 1.0)

const Levy = Alpha

##############################################################################
#
# Arcsine distribution
#
# REFERENCES: Using definition from Devroye, IX.7
#  This definition differs from definitions on Wikipedia and other sources,
#  where distribution is over [0, 1] rather than [-1, 1].
#
# TODO: Implement kurtosis and entropy.
#
##############################################################################

immutable Arcsine <: ContinuousUnivariateDistribution
end

function cdf(d::Arcsine, x::Number)
  if x < -1.
    return 0.
  elseif x > 1.
    return 1.
  else
    return (2. / pi) * asin(sqrt((x + 1.) / 2.))
  end
end
function insupport(d::Arcsine, x::Number)
  if -1. <= x <= 1.
    return true
  else
    return false
  end
end
mean(d::Arcsine) = 0.0
median(d::Arcsine) = 0.0
function pdf(d::Arcsine, x::Number)
  if insupport(d, x)
    1. / (pi * sqrt(1. - x^2))
  else
    0.
  end
end
quantile(d::Arcsine, p::Real) = 2. * sin((pi / 2.) * p)^2 - 1.
rand(d::Arcsine) = sin(2. * pi * rand())
skewness(d::Arcsine) = 0.
var(d::Arcsine) = 1./2.

##############################################################################
#
# Bernoulli distribution
#
# REFERENCES: Wasserman, "All of Statistics"
#
# TODO: Test kurtosis, skewness
#
# NOTES: Fails CDF/Quantile matching test
#
##############################################################################

# TODO: Move these two methods into Base
xlogx(x::Real) = x == 0.0 ? 0.0 : x * log(x)
xlogxdmu(x::Real, mu::Real) = x == 0.0 ? 0.0 : x * log(x/mu)
function entropy(p::Vector)
    e = 0.0
    for p_i in p
        if p_i < 0.
            error("Probabilities must lie in [0, 1]")
        end
        e -= xlogx(p_i)
    end
    return e
end

immutable Bernoulli <: DiscreteUnivariateDistribution
    prob::Float64
    Bernoulli(p::Real) = 0. <= p <= 1. ? new(float64(p)) : error("prob must be in [0,1]")
end
Bernoulli() = Bernoulli(0.5)

cdf(d::Bernoulli, q::Real) = q < 0. ? 0. : (q >= 1. ? 1. : 1. - d.prob)
function devresid(d::Bernoulli, y::Real, mu::Real, wt::Real)
    2wt*(xlogxdmu(y,mu) + xlogxdmu(1.-y,1.-mu))
end
function devresid(d::Bernoulli, y::Vector{Float64}, mu::Vector{Float64}, wt::Vector{Float64})
    [2wt[i]*(xlogxdmu(y[i],mu[i]) + xlogxdmu(1-y[i],1-mu[i])) for i in 1:length(y)]
end
entropy(d::Bernoulli) = -xlogx(1.0 - d.prob) - xlogx(d.prob)
insupport(d::Bernoulli, x::Number) = (x == 0) || (x == 1)
kurtosis(d::Bernoulli) = 1 / var(d) - 6.
logpdf( d::Bernoulli, mu::Real, y::Real) = y==0 ? log(1. - mu): (y == 1 ? log(mu): -Inf)
mean(d::Bernoulli) = d.prob
median(d::Bernoulli) = d.prob < 0.5 ? 0.0 : 1.0
function modes(d::Bernoulli)
  if d.prob < 0.5
    [0]
  elseif d.prob == 0.5
    [0, 1]
  else
    [1]
  end
end
mustart(d::Bernoulli,  y::Real, wt::Real) = (wt * y + 0.5) / (wt + 1)
pdf(d::Bernoulli, x::Real) = x == 0 ? (1 - d.prob) : (x == 1 ? d.prob : 0)
quantile(d::Bernoulli, p::Real) = 0 < p < 1 ? (p <= (1. - d.prob) ? 0 : 1) : NaN
rand(d::Bernoulli) = rand() > d.prob ? 0 : 1
skewness(d::Bernoulli) = (1-2d.prob)/std(d)
var(d::Bernoulli, mu::Real) = max(eps(), mu*(1. - mu))
var(d::Bernoulli) = d.prob * (1. - d.prob)

##############################################################################
#
# Beta distribution
#
##############################################################################

immutable Beta <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    Beta(a, b) = a > 0 && b > 0 ? new(float64(a), float64(b)) : error("Both alpha and beta must be positive")
end
Beta(a) = Beta(a, a)                    # symmetric in [0,1]
Beta()  = Beta(1)                       # uniform
@_jl_dist_2p Beta beta
function entropy(d::Beta)
  o = lbeta(d.alpha, d.beta)
  o = o - (d.alpha - 1.0) * digamma(d.alpha)
  o = o - (d.beta - 1.0) * digamma(d.beta)
  o = o + (d.alpha + d.beta - 2.0) * digamma(d.alpha + d.beta)
  return o
end
mean(d::Beta) = d.alpha / (d.alpha + d.beta)
function modes(d::Beta)
  if d.alpha > 1.0 && d.beta > 1.0
    return [(d.alpha - 1.) / (d.alpha + d.beta - 2.)]
  else
    error("Beta distribution with (a, b) has no modes")
  end
end
var(d::Beta) = (ab = d.alpha + d.beta; d.alpha * d.beta /(ab * ab * (ab + 1.)))
function skewness(d::Beta)
  d = 2.0 * (d.beta - d.alpha) * sqrt(d.alpha + d.beta + 1.0)
  n = (d.alpha + d.beta + 2.0) * sqrt(d.alpha * d.beta)
  return d / n
end
function kurtosis(d::Beta)
  a, b = d.alpha, d.beta
  d = 6.0 * ((a - b)^2 * (a + b + 1.0) - a * b * (a + b + 2.0))
  n = a * b * (a + b + 2.0) * (a + b + 3.0)
  return d / n
end
rand(d::Beta) = (u = rand(Gamma(d.alpha)); u / (u + rand(Gamma(d.beta))))
rand(d::Beta, dims::Dims) = (u = rand(Gamma(d.alpha), dims); u ./ (u + rand(Gamma(d.beta), dims)))
rand!(d::Beta, A::Array{Float64}) = (A[:] = randbeta(d.alpha, d.beta, size(A)))
insupport(d::Beta, x::Number) = real_valued(x) && 0 < x < 1

##############################################################################
#
# BetaPrime distribution
#
# REFERENCES: Forbes et al. "Statistical Distributions"
#
##############################################################################

immutable BetaPrime <: ContinuousUnivariateDistribution
  alpha::Float64
  beta::Float64
  function BetaPrime(a::Real, b::Real)
    if a > 0.0 && b > 0.0
      new(float64(a), float64(b))
    else
      error("Both alpha and beta must be positive")
    end
  end
end
BetaPrime() = BetaPrime(2.0, 1.0)

cdf(d::BetaPrime, q::Real) = inc_beta(q / 1.0 + q, d.alpha, d.beta)
insupport(d::BetaPrime, x::Number) = real_valued(x) && x > 0 ? true : false
function mean(d::BetaPrime)
  if d.beta > 1.0
    d.alpha / (d.beta + 1.0)
  else
    error("mean not defined when beta <= 1")
  end
end
function pdf(d::BetaPrime, x::Real)
  a, b = d.alpha, d.beta
  (x^(a - 1.0) * (10. + x)^(-(a + b))) / beta(a, b)
end
rand(d::BetaPrime) = 1 / randbeta(d.alpha, d.beta)

##############################################################################
#
# Binomial distribution
#
##############################################################################

immutable Binomial <: DiscreteUnivariateDistribution
    size::Int
    prob::Float64
    Binomial(n, p) = n <= 0 ?  error("size must be positive") : (0. <= p <= 1. ? new(int(n), float64(p)) : error("prob must be in [0,1]"))
end
Binomial(size) = Binomial(size, 0.5)
Binomial()     = Binomial(1, 0.5)
@_jl_dist_2p Binomial binom
kurtosis(d::Binomial) = (1.0 - 2.0 * d.prob * (1. - d.prob)) / var(d)
mean(d::Binomial)     = d.size * d.prob
modes(d::Binomial) = iround([d.size * d.prob])
var(d::Binomial)      = d.size * d.prob * (1. - d.prob)
skewness(d::Binomial) = (1-2d.prob)/std(d)
kurtosis(d::Binomial) = (1-2d.prob*(1-d.prob))/var(d)
insupport(d::Binomial, x::Number) = integer_valued(x) && 0 <= x <= d.size

##############################################################################
#
# Cauchy distribution
#
##############################################################################

immutable Cauchy <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    Cauchy(l, s) = s > 0 ? new(float64(l), float64(s)) : error("scale must be positive")
end
Cauchy(l) = Cauchy(l, 1.0)
Cauchy()  = Cauchy(0.0, 1.0)
@_jl_dist_2p Cauchy cauchy
mean(d::Cauchy)     = NaN
var(d::Cauchy)      = NaN
skewness(d::Cauchy) = NaN
kurtosis(d::Cauchy) = NaN
insupport(d::Cauchy, x::Number) = real_valued(x) && isfinite(x)

immutable Chi <: ContinuousUnivariateDistribution
    df::Float64
end

immutable Chisq <: ContinuousUnivariateDistribution
    df::Float64      # non-integer degrees of freedom are meaningful
    Chisq(d) = d > 0 ? new(float64(d)) : error("df must be positive")
end
@_jl_dist_1p Chisq chisq
mean(d::Chisq)     = d.df
var(d::Chisq)      = 2d.df
skewness(d::Chisq) = sqrt(8/d.df)
kurtosis(d::Chisq) = 12/d.df
## rand - the distribution chi^2(df) is 2*gamma(df/2)
## for integer n, a chi^2(n) is the sum of n squared standard normals
rand(d::Chisq) = d.df == 1 ? randn()^2 : 2.rand(Gamma(d.df/2.))
function rand!(d::Chisq, A::Array{Float64})
    if d.df == 1
        for i in 1:length(A)
            A[i] = randn()^2
            end
        return A
    end
    dpar = d.df >= 2 ? d.df/2. - 1.0/3.0 : error("require degrees of freedom df >= 2")
    cpar = 1.0/sqrt(9.0dpar)
    for i in 1:length(A) A[i] = 2.randg2(dpar,cpar) end
    A
end
insupport(d::Chisq, x::Number) =  real_valued(x) && isfinite(x) && 0 <= x

immutable DiscreteUniform <: DiscreteUnivariateDistribution
  a::Int
  b::Int
  function DiscreteUniform(a::Real, b::Real)
    if a < b
      new(int(a), int(b))
    else
      error("a must be less than b")
    end
  end
end
DiscreteUniform(b::Int) = DiscreteUniform(0, b)
DiscreteUniform() = DiscreteUniform(0, 1)

insupport(d::DiscreteUniform, x::Number) = isinteger(x) && d.a <= x <= d.b
function kurtosis(d::DiscreteUniform)
  n = d.b - d.a + 1.
  -(6. / 5.) * (n^2 + 1.)/(n^2 - 1.)
end
mean(d::DiscreteUniform) = (d.a + d.b) / 2.
median(d::DiscreteUniform) = (d.a + d.b) / 2.
pdf(d::DiscreteUniform, x::Real) =
  insupport(d, x) ? (1.0 / (d.b - d.a + 1.)) : 0.0
function quantile(d::DiscreteUniform, k::Real)
  if k < d.a
    0.0
  elseif <= d.b
    (floor(k) - d.a + 1.) / (d.b - d.a + 1.)
  else
    1.0
  end
end
function rand(d::DiscreteUniform)
  d.a + rand(0:(d.b - d.a))
end
skewness(d::DiscreteUniform) = 0.0
var(d::DiscreteUniform) = ((d.b - d.a + 1.0)^2 - 1.0) / 12.0

immutable Erlang <: ContinuousUnivariateDistribution
    shape::Float64
    rate::Float64
end

immutable Exponential <: ContinuousUnivariateDistribution
    scale::Float64                      # note: scale not rate
    Exponential(sc) = sc > 0 ? new(float64(sc)) : error("scale must be positive")
end
Exponential() = Exponential(1.)
mean(d::Exponential)     = d.scale
median(d::Exponential)   = d.scale * log(2.)
var(d::Exponential)      = d.scale * d.scale
skewness(d::Exponential) = 2.
kurtosis(d::Exponential) = 6.
cdf(d::Exponential, q::Real) = q <= 0. ? 0. : -expm1(-q/d.scale)
function logcdf(d::Exponential, q::Real)
    q <= 0. ? -Inf : (qs = -q/d.scale; qs > log(0.5) ? log(-expm1(qs)) : log1p(-exp(qs)))
end
function ccdf(d::Exponential, q::Real)
    q <= 0. ? 1. : exp(-q/d.scale)
end
function logccdf(d::Exponential, q::Real)
    q <= 0. ? 0. : -q/d.scale
end
function pdf(d::Exponential, x::Real)
    x <= 0. ? 0. : exp(-x/d.scale) / d.scale
end
function logpdf(d::Exponential, x::Real)
    x <= 0. ? -Inf : (-x/d.scale) - log(d.scale)
end
function quantile(d::Exponential, p::Real)
    0. <= p <= 1. ? -d.scale * log1p(-p) : NaN
end
function invlogcdf(d::Exponential, lp::Real)
    lp <= 0. ? -d.scale * (lp > log(0.5) ? log(-expm1(lp)) : log1p(-exp(lp))) : NaN
end
function cquantile(d::Exponential, p::Real)
    0. <= p <= 1. ? -d.scale * log(p) : NaN
end
function invlogccdf(d::Exponential, lp::Real)
    lp <= 0. ? -d.scale * lp : NaN
end
rand(d::Exponential)                     = d.scale * Random.randmtzig_exprnd()
rand!(d::Exponential, A::Array{Float64}) = d.scale * Random.randmtzig_fill_exprnd!(A)
insupport(d::Exponential, x::Number) = real_valued(x) && isfinite(x) && 0 <= x

immutable FDist <: ContinuousUnivariateDistribution
    ndf::Float64
    ddf::Float64
    FDist(d1,d2) = d1 > 0 && d2 > 0 ? new(float64(d1), float64(d2)) : error("Both numerator and denominator degrees of freedom must be positive")
end
@_jl_dist_2p FDist f
mean(d::FDist) = 2 < d.ddf ? d.ddf/(d.ddf - 2) : NaN
var(d::FDist)  = 4 < d.ddf ? 2d.ddf^2*(d.ndf+d.ddf-2)/(d.ndf*(d.ddf-2)^2*(d.ddf-4)) : NaN
insupport(d::FDist, x::Number) = real_valued(x) && isfinite(x) && 0 <= x

immutable Gamma <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    Gamma(sh,sc) = sh > 0 && sc > 0 ? new(float64(sh), float64(sc)) : error("Both shape and scale must be positive")
end
Gamma(sh) = Gamma(sh, 1.)
Gamma()   = Gamma(1., 1.)               # Standard exponential distribution
@_jl_dist_2p Gamma gamma
mean(d::Gamma)     = d.shape * d.scale
var(d::Gamma)      = d.shape * d.scale * d.scale
skewness(d::Gamma) = 2/sqrt(d.shape)
## rand()
# A simple method for generating gamma variables - Marsaglia and Tsang (2000)
# http://www.cparity.com/projects/AcmClassification/samples/358414.pdf
# Page 369
# basic simulation loop for pre-computed d and c
function randg2(d::Float64, c::Float64) 
    while true
        x = v = 0.0
        while v <= 0.0
            x = randn()
            v = 1.0 + c*x
        end
        v = v^3
        U = rand()
        x2 = x^2
        if U < 1.0-0.331*x2^2 || log(U) < 0.5*x2+d*(1.0-v+log(v))
            return d*v
        end
    end
end
function rand(d::Gamma)
    dpar = (d.shape <= 1. ? d.shape + 1 : d.shape) - 1.0/3.0
    d.scale*randg2(dpar, 1.0/sqrt(9.0dpar)) * (d.shape > 1. ? 1. : rand()^(1./d.shape))
end
function rand!(d::Gamma, A::Array{Float64})
    dpar = (d.shape <= 1. ? d.shape + 1 : d.shape) - 1.0/3.0
    cpar = 1.0/sqrt(9.0dpar)
    for i in 1:length(A) A[i] = randg2(dpar, cpar) end
    if d.shape <= 1.
        ainv = 1./d.shape
        for i in 1:length(A) A[i] *= rand()^ainv end
    end
    d.scale*A
end
insupport(d::Gamma, x::Number) = real_valued(x) && isfinite(x) && 0 <= x

###########################################################################
## InvertedGamma distribution, cf Bayesian Theory (Barnardo & Smith) p 119
## Note that B&S parametrize in terms of shape/rate, but this is uses
## shape/scale so that it is consistent with the implementation of Gamma
###########################################################################
immutable InvertedGamma <: ContinuousUnivariateDistribution
    shape::Float64 # location
    scale::Float64 # scale
    InvertedGamma(sh,sc) = sh > 0 && sc > 0 ? new(float64(sh), float64(sc)) : error("Both shape and scale must be positive")
end
mean(d::InvertedGamma) = d.shape > 1. ? (1. / d.scale) / (d.shape - 1.) : error("Expectation only defined if shape > 1")
var(d::InvertedGamma) = d.shape > 2. ? (1. / d.scale)^2 / ((d.shape - 1.)^2 * (d.shape - 2.)) : error("Variance only defined if shape > 2")
function rand(d::InvertedGamma)
   1. / rand(Gamma(d.shape, d.scale))
end
function rand!(d::InvertedGamma, A::Array{Float64})
    A = rand!(Gamma(d.shape, d.scale), A)
    1. / A
end
insupport(d::InvertedGamma, x::Number) = real_valued(x) && isfinite(x) && 0 <= x
function logpdf(d::InvertedGamma, x::Real)
    -(d.shape * log(d.scale)) - lgamma(d.shape) - ((d.shape + 1) * log(x)) - 1./(d.scale * x)
end
function pdf(d::InvertedGamma, x::Real)
    exp(logpdf(d, x))
end
function cdf(d::InvertedGamma, x::Real)
    1. - cdf(Gamma(d.shape, d.scale), 1./x)
end
function quantile(d::InvertedGamma, p::Real)
    1. / quantile(Gamma(d.shape, d.scale), 1 - p)
end
modes(d::InvertedGamma) = [(1./d.scale) / (d.shape + 1)]

immutable Geometric <: DiscreteUnivariateDistribution
    # In the form of # of failures before the first success
    prob::Float64
    Geometric(p) = 0 < p < 1 ? new(float64(p)) : error("prob must be in (0,1)")
end
Geometric() = Geometric(0.5)            # Flips of a fair coin
@_jl_dist_1p Geometric geom
mean(d::Geometric)     = (1-d.prob)/d.prob
var(d::Geometric)      = (1-d.prob)/d.prob^2
skewness(d::Geometric) = (2-d.prob)/sqrt(1-d.prob)
kurtosis(d::Geometric) = 6+d.prob^2/(1-d.prob)
function cdf(d::Geometric, q::Real)
    q < 0. ? 0. : -expm1(log1p(-d.prob) * (floor(q) + 1.))
end
function ccdf(d::Geometric, q::Real)
    q < 0. ? 1. : exp(log1p(-d.prob) * (floor(q + 1e-7) + 1.))
end
insupport(d::Geometric, x::Number) = integer_valued(x) && 0 <= x

immutable HyperGeometric <: DiscreteUnivariateDistribution
    ns::Float64                         # number of successes in population
    nf::Float64                         # number of failures in population
    n::Float64                          # sample size
    function HyperGeometric(s,f,n)
        s = 0 <= s && int(s) == s ? int(s) : error("ns must be a non-negative integer")
        f = 0 <= f && int(f) == f ? int(f) : error("nf must be a non-negative integer")        
        n = 0 < n <= (s+f) && int(n) == n ? new(float64(s), float64(f), float64(n)) : error("n must be a positive integer <= (ns + nf)")
    end
end
@_jl_dist_3p HyperGeometric hyper
mean(d::HyperGeometric) = d.n*d.ns/(d.ns+d.nf)
var(d::HyperGeometric)  = (N=d.ns+d.nf; p=d.ns/N; d.n*p*(1-p)*(N-d.n)/(N-1))
insupport(d::HyperGeometric, x::Number) = integer_valued(x) && 0 <= x <= d.n && (d.n - d.nf) <= x <= d.ns

immutable Laplace <: ContinuousUnivariateDistribution
  location::Float64
  scale::Float64
  function Laplace(l::Real, s::Real)
    if s > 0.0
      new(float64(l), float64(s))
    else
      error("scale must be positive")
    end
  end
end
Laplace(location::Real) = Laplace(location, 1.0)
Laplace() = Laplace(0.0, 1.0)

const Biexponential = Laplace

function cdf(d::Laplace, q::Real)
  0.5 * (1.0 + sign(q - d.location) * (1.0 - exp(-abs(q - d.location) / d.scale)))
end
insupport(d::Laplace, x::Number) = real_valued(x) && isfinite(x)
kurtosis(d::Laplace) = 3.0
mean(d::Laplace) = d.location
median(d::Laplace) = d.location
function pdf(d::Laplace, x::Real)
  (1.0 / (2.0 * d.scale)) * exp(-abs(x - d.location) / d.scale)
end
function logpdf(d::Laplace, x::Real)
  -log(2.0 * d.scale) - abs(x - d.location) / d.scale
end
function quantile(d::Laplace, p::Real)
  d.location - d.scale * sign(p - 0.5) * log(1.0 - 2.0 * abs(p - 0.5))
end
# Need to see whether other RNG strategies are more efficient:
# (1) Difference of two Exponential(1/b) variables
# (2) Ratio of logarithm of two Uniform(0.0, 1.0) variables
function rand(d::Laplace)
  u = rand() - 0.5
  return d.location - d.scale * sign(u) * log(1.0 - 2.0 * abs(u))
end
skewness(d::Laplace) = 0.0
std(d::Laplace) = sqrt(2.0) * d.scale
var(d::Laplace) = 2.0 * d.scale^2

immutable Logistic <: ContinuousUnivariateDistribution
    location::Real
    scale::Real
    Logistic(l, s) = s > 0 ? new(float64(l), float64(s)) : error("scale must be positive")
end
Logistic(l) = Logistic(l, 1)
Logistic()  = Logistic(0, 1)
@_jl_dist_2p Logistic logis
mean(d::Logistic)     = d.location
median(d::Logistic)   = d.location
var(d::Logistic)      = (pi*d.scale)^2/3.
std(d::Logistic)      = pi*d.scale/sqrt(3.)
skewness(d::Logistic) = 0.
kurtosis(d::Logistic) = 1.2
insupport(d::Logistic, x::Number) = real_valued(x) && isfinite(x)

immutable logNormal <: ContinuousUnivariateDistribution
    meanlog::Float64
    sdlog::Float64
    logNormal(ml,sdl) = sdl > 0 ? new(float64(ml), float64(sdl)) : error("sdlog must be positive")
end
logNormal(ml) = logNormal(ml, 1)
logNormal()   = logNormal(0, 1)
@_jl_dist_2p logNormal lnorm
mean(d::logNormal) = exp(d.meanlog + d.sdlog^2/2)
var(d::logNormal)  = (sigsq=d.sdlog^2; (exp(sigsq) - 1)*exp(2d.meanlog+sigsq))
insupport(d::logNormal, x::Number) = real_valued(x) && isfinite(x) && 0 < x

immutable MixtureModel <: Distribution
  components::Vector # Vector should be able to contain any type of
                     # distribution with comparable support
  probs::Vector{Float64}
  function MixtureModel(c::Vector, p::Vector{Float64})
    if length(c) != length(p)
      error("components and probs must have the same number of elements")
    end
    sump = 0.0
    for i in 1:length(p)
      if p[i] < 0.0
        error("MixtureModel: probabilities must be non-negative")
      end
      sump += p[i]
    end
    new(c, p ./ sump)
  end
end

function pdf(d::MixtureModel, x::Any)
  p = 0.0
  for i in 1:length(d.components)
    p += pdf(d.components[i], x) * d.probs[i]
  end
  return p
end
function rand(d::MixtureModel)
  i = rand(Categorical(d.probs))
  rand(d.components[i])
end
function mean(d::MixtureModel)
  m = 0.0
  for i in 1:length(d.components)
    m += mean(d.components[i]) * d.probs[i]
  end
  return m
end
function var(d::MixtureModel)
  m = 0.0
  for i in 1:length(d.components)
    m += var(d.components[i]) * d.probs[i]^2
  end
  return m
end

immutable MultivariateNormal <: ContinuousMultivariateDistribution
  mean::Vector{Float64}
  covchol::Cholesky{Float64}
  function MultivariateNormal(m, c)
    if length(m) == size(c, 1) == size(c, 2)
      new(m, c)
    else
      error("Dimensions of mean vector and covariance matrix do not match")
    end
  end
end
MultivariateNormal(mean::Vector{Float64}, cov::Matrix{Float64}) = MultivariateNormal(mean, cholfact(cov))
MultivariateNormal(mean::Vector{Float64}) = MultivariateNormal(mean, eye(length(mean)))
MultivariateNormal(cov::Matrix{Float64}) = MultivariateNormal(zeros(size(cov, 1)), cov)
MultivariateNormal() = MultivariateNormal(zeros(2), eye(2))

mean(d::MultivariateNormal) = d.mean
var(d::MultivariateNormal) = (U = d.covchol[:U]; U'U)
function rand(d::MultivariateNormal)
  z = randn(length(d.mean))
  return d.mean + d.covchol[:U]'z
end
function rand!(d::MultivariateNormal, X::Matrix)
  k = length(mean(d))
  m, n = size(X)
  if m == k return d.covchol[:U]'randn!(X) + d.mean[:,ones(Int,n)] end
  if n == k return randn!(X) * d.covchol[:U] + d.mean'[ones(Int,m),:] end
  error("Wrong dimensions")
end
function logpdf{T <: Real}(d::MultivariateNormal, x::Vector{T})
  k = length(d.mean)
  u = x - d.mean
  z = d.covchol \ u  # This is equivalent to inv(cov) * u, but much faster
  return -0.5 * k * log(2.0pi) - sum(log(diag(d.covchol[:U]))) - 0.5 * dot(u,z)
end
pdf{T <: Real}(d::MultivariateNormal, x::Vector{T}) = exp(logpdf(d, x))
function cdf{T <: Real}(d::MultivariateNormal, x::Vector{T})
  k = length(d.mean)
  if k > 3; error("Dimension larger than three is not supported yet"); end
  stddev = sqrt(diag(var(d)))
  z = (x - d.mean) ./ stddev
  C = diagmm(d.covchol[:U], 1.0 / stddev)
  C = C'C
  if k == 3
    return tvtcdf(0, z, C[[2, 3, 6]])
  elseif k == 2
    return bvtcdf(0, z[1], z[2], C[2])
  else
    return cdf(Normal(), z[1])
  end
end


#########################################################
## Wishart Distribution
## Parameters nu and S such that E(X) = nu * S 
## See the rwish and dwish implementation in R's MCMCPack
## This parametrization differs from Barnardo & Smith p 435
## in this way: (nu, S) = (2*alpha, .5*beta^-1) 
#########################################################
immutable Wishart <: ContinuousMatrixDistribution
  nu::Float64
  Schol::Cholesky{Float64}
  function Wishart(n::Float64, Sc::Cholesky{Float64})
    if n > (size(Sc, 1) - 1.)
      new(n, Sc)
    else
      error("Wishart parameters must be df > p - 1")
    end
  end
end
Wishart(nu::Float64, S::Matrix{Float64}) = Wishart(nu, cholfact(S))
Wishart(nu::Int64, S::Matrix{Float64}) = Wishart(convert(Float64, nu), S)
Wishart(nu::Int64, Schol::Cholesky{Float64}) = Wishart(convert(Float64, nu), Schol)
mean(w::Wishart) = w.nu * w.Schol[:U]' * w.Schol[:U]
var(w::Wishart) = "TODO"

function rand(w::Wishart)
  p = size(w.Schol, 1)
  X = zeros(p,p)
  for ii in 1:p
    X[ii,ii] = sqrt(rand(Chisq(w.nu - ii + 1)))
  end
  if p > 1
    for col in 2:p
      for row in 1:(col - 1)
        X[row,col] = randn()
      end
    end
  end
  Z = X * w.Schol[:U]
  return Z' * Z
end

function insupport(W::Wishart, X::Matrix{Float64})
 return size(X,1) == size(X,2) && isApproxSymmmetric(X) && size(X,1) == size(W.Schol,1) && hasCholesky(X)
end

function logpdf(W::Wishart, X::Matrix{Float64})
  if !insupport(W, X)
    return -Inf
  else
    p = size(X,1)
    logd::Float64 = - (W.nu * p / 2. * log(2) + W.nu / 2. * log(det(W.Schol)) + lpgamma(p, W.nu/2.))
    logd += 0.5 * (W.nu - p - 1.) * logdet(X)
    logd -= 0.5 * trace(W.Schol \ X) #logd -= 0.5 * trace(inv(W.Schol)*X)
    return logd
  end
end

## multivariate gamma / partial gamma function
function lpgamma(p::Int64, a::Float64)
  res::Float64 = p * (p - 1.) / 4. * log(pi)
  for ii in 1:p
    res += lgamma(a + (1. - ii)/2.)
  end
  return res
end

function pdf(W::Wishart, X::Matrix{Float64})
  return exp(logpdf(W, X))
end


######################################################
## Inverse Wishart Distribution
## Parametrized such that E(X) = Psi / (nu - p - 1)
## See the riwish and diwish function of R's MCMCpack
######################################################
immutable InverseWishart <: ContinuousMatrixDistribution
  nu::Float64
  Psichol::Cholesky{Float64}
  function InverseWishart(n::Float64, Pc::Cholesky{Float64})
    if n > (size(Pc, 1) - 1)
      new(n, Pc)
    else
      error("Inverse Wishart parameters must be df > p - 1")
    end
  end
end
InverseWishart(nu::Float64, Psi::Matrix{Float64}) = InverseWishart(nu, cholfact(Psi))
InverseWishart(nu::Int64, Psi::Matrix{Float64}) = InverseWishart(convert(Float64, nu), Psi)
InverseWishart(nu::Int64, Psichol::Cholesky{Float64}) = InverseWishart(convert(Float64, nu), Psichol)
mean(IW::InverseWishart) =  IW.nu > (size(IW.Psichol, 1) + 1) ? 1/(IW.nu - size(IW.Psichol, 1) - 1) * IW.Psichol[:U]' * IW.Psichol[:U] : "mean only defined for nu > p + 1"
var(IW::InverseWishart) = "TODO"

function rand(IW::InverseWishart)
  ## rand(Wishart(nu, Psi^-1))^-1 is an sample from an inverse wishart(nu, Psi)
  return inv(rand(Wishart(IW.nu, inv(IW.Psichol))))
  ## there is actually some wacky behavior here where inv of the Cholesky returns the 
  ## inverse of the original matrix, in this case we're getting Psi^-1 like we want
end

function rand!(IW::InverseWishart, X::Array{Matrix{Float64}})
  Psiinv = inv(IW.Psichol)
  W = Wishart(IW.nu, Psiinv)
  X = rand!(W, X)
  for i in 1:length(X)
    X[i] = inv(X[i])
  end
  return X
end

function insupport(IW::InverseWishart, X::Matrix{Float64})
  return size(X,1) == size(X,2) && isApproxSymmmetric(X) && size(X,1) == size(IW.Psichol,1) && hasCholesky(X)
end

function pdf(IW::InverseWishart, X::Matrix{Float64})
  return exp(logpdf(IW, X))
end

function logpdf(IW::InverseWishart, X::Matrix{Float64})
  if !insupport(IW, X)
    return -Inf
  else
    p = size(X,1)
    logd::Float64 = - ( IW.nu * p / 2. * log(2) + lpgamma(p, IW.nu / 2.) - IW.nu / 2. * log(det(IW.Psichol)))
    logd -= 0.5 * (IW.nu + p + 1) * logdet(X)
##    logd -= 0.5 * trace(IW.Psichol[:U]' * IW.Psichol[:U] * inv(X))
    logd -= 0.5 * trace(X \ (IW.Psichol[:U]' * IW.Psichol[:U]))
    return logd
  end
end

## because X==X' keeps failing due to floating point nonsense
function isApproxSymmmetric(a::Matrix{Float64})
  tmp = true
  for j in 2:size(a,1)
    for i in 1:j-1
      tmp &= abs(a[i,j] - a[j,i]) < 1e-8
    end
  end
  return tmp
end

## because isposdef keeps giving the wrong answer for samples from Wishart and InverseWisharts
function hasCholesky(a::Matrix{Float64})
  try achol = cholfact(a)
  catch e
    return false
  end
  return true
end

## NegativeBinomial is the distribution of the number of failures
## before the size'th success in a sequence of Bernoulli trials.
## We do not enforce integer size, as the distribution is well defined
## for non-integers, and this can be useful for e.g. overdispersed
## discrete survival times.
immutable NegativeBinomial <: DiscreteUnivariateDistribution
    size::Float64
    prob::Float64
    NegativeBinomial(s,p) = 0 < p <= 1 ? (s >= 0 ? new(float64(s),float64(p)) : error("size must be non-negative")) : error("prob must be in (0,1]")
end
@_jl_dist_2p NegativeBinomial nbinom
insupport(d::NegativeBinomial, x::Number) = integer_valued(x) && 0 <= x

immutable NoncentralBeta <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    ncp::Float64
    NoncentralBeta(a,b,nc) = a > 0 && b > 0 && nc >= 0 ? new(float64(a),float64(b),float64(nc)) : error("alpha and beta must be > 0 and ncp >= 0")
end
@_jl_dist_3p NoncentralBeta nbeta

immutable NoncentralChisq <: ContinuousUnivariateDistribution
    df::Float64
    ncp::Float64
    NoncentralChisq(d,nc) = d >= 0 && nc >= 0 ? new(float64(d),float64(nc)) : error("df and ncp must be non-negative")
end
@_jl_dist_2p NoncentralChisq nchisq
insupport(d::NoncentralChisq, x::Number) = real_valued(x) && isfinite(x) && 0 < x

immutable NoncentralF <: ContinuousUnivariateDistribution
    ndf::Float64
    ddf::Float64
    ncp::Float64
    NoncentralF(n,d,nc) = n > 0 && d > 0 && nc >= 0 ? new(float64(n),float64(d),float64(nc)) : error("ndf and ddf must be > 0 and ncp >= 0")
end
@_jl_dist_3p NoncentralF nf
insupport(d::logNormal, x::Number) = real_valued(x) && isfinite(x) && 0 <= x

immutable NoncentralT <: ContinuousUnivariateDistribution
    df::Float64
    ncp::Float64
    NoncentralT(d,nc) = d >= 0 && nc >= 0 ? new(float64(d),float64(nc)) : error("df and ncp must be non-negative")
end
@_jl_dist_2p NoncentralT t
insupport(d::NoncentralT, x::Number) = real_valued(x) && isfinite(x)

immutable Normal <: ContinuousUnivariateDistribution
    mean::Float64
    std::Float64
    Normal(mu, sd) = sd > 0 ? new(float64(mu), float64(sd)) : error("std must be positive")
end
Normal(mu) = Normal(mu, 1.0)
Normal() = Normal(0.0, 1.0)
const Gaussian = Normal
@_jl_dist_2p Normal norm
mean(d::Normal) = d.mean
median(d::Normal) = d.mean
var(d::Normal) = d.std^2
skewness(d::Normal) = 0.
kurtosis(d::Normal) = 0.
## redefine common methods
rand(d::Normal) = d.mean + d.std * randn()
insupport(d::Normal, x::Number) = real_valued(x) && isfinite(x)

immutable Pareto <: ContinuousUnivariateDistribution
  scale::Float64
  shape::Float64
  function Pareto(sc, sh)
    if sc > 0.0 && sh > 0.0
      new(float64(sc), float64(sh))
    else
      error("shape and scale must be positive")
    end
  end
end
Pareto(scale::Float64) = Pareto(scale, 1.)
Pareto() = Pareto(1., 1.)

cdf(d::Pareto, q::Real) = q >= d.scale ? 1. - (d.scale / q)^d.shape : 0.
insupport(d::Pareto, x::Number) = real_valued(x) && isfinite(x) && x > d.scale
function kurtosis(d::Pareto)
  a = d.shape
  if a > 4.
    (6. * (a^3 + a^2 - 6.a - 2)) / (a * (a - 3.) * (a - 4.))
  else
    error("Kurtosis undefined for Pareto w/ shape <= 4")
  end
end
mean(d::Pareto) = d.shape <= 1. ? Inf : (d.scale * d.shape) / (d.scale - 1.)
median(d::Pareto) = d.scale * 2.^(d.shape)
function pdf(d::Pareto, q::Real)
  if q >= d.scale
    return (d.shape * d.scale^d.shape) / (q^(d.shape + 1.))
  else
    return 0.
  end
end
quantile(d::Pareto, p::Real) = d.scale / (1. - p)^(1. / d.shape)
rand(d::Pareto) = d.shape / (rand()^(1. / d.scale))
function skewness(d::Pareto)
  a = d.shape
  if a > 3.
    ((2. * (1. + a)) / (a - 3.)) * sqrt((a - 2.)/a)
  else
    error("Skewness undefined for Pareto w/ shape <= 3")
  end
end
function var(d::Pareto)
  if d.scale < 2.
    return Inf
  else
    return (d.shape^2 * d.scale) / ((d.scale - 1.)^2 * (d.scale - 2.))
  end
end

immutable Poisson <: DiscreteUnivariateDistribution
    lambda::Float64
    Poisson(l) = l > 0 ? new(float64(l)) : error("lambda must be positive")
end
Poisson() = Poisson(1)
@_jl_dist_1p Poisson pois
devresid(d::Poisson,  y::Real, mu::Real, wt::Real) = 2wt*(xlogxdmu(y,mu) - (y-mu))
function devresid(d::Poisson, y::Vector{Float64}, mu::Vector{Float64}, wt::Vector{Float64})
    [2wt[i]*(xlogxdmu(y[i],mu[i]) - (y[i]-mu[i])) for i in 1:length(y)]
end
insupport(d::Poisson, x::Number) = integer_valued(x) && 0 <= x
logpdf(  d::Poisson, mu::Real, y::Real) = ccall((:dpois, Rmath),Float64,(Float64,Float64,Int32),y,mu,1)
mean(d::Poisson) = d.lambda
mustart( d::Poisson,  y::Real, wt::Real) = y + 0.1
var(     d::Poisson, mu::Real) = mu
var(d::Poisson) = d.lambda

##############################################################################
#
# Rayleigh distribution from Distributions Handbook
#
##############################################################################

immutable Rayleigh <: ContinuousUnivariateDistribution
  scale::Float64
  function Rayleigh(s)
    if s > 0.0
      new(float64(s))
    else
      error("scale must be positive")
    end
  end
end
Rayleigh() = Rayleigh(1.0)

insupport(d::Rayleigh, x::Number) = real_valued(x) && isfinite(x) && x > 0.
kurtosis(d::Rayleigh) = d.scale^4 * (8. - ((3. * pi^2) / 4.))
mean(d::Rayleigh) = d.scale * sqrt(pi / 2.)
median(d::Rayleigh) = d.scale * sqrt(2. * log(2.))
function pdf(d::Rayleigh, x::Real)
  if insupport(d, x)
    return (x / (d.scale^2)) * exp(-(x^2)/(2. * (d.scale^2)))
  else
    return 0.
  end
end
rand(d::Rayleigh) = d.scale * sqrt(-2.log(rand()))
skewness(d::Rayleigh) = d.scale^3 * (pi - 3.) * sqrt(pi / 2.)
var(d::Rayleigh) = d.scale^2 * (2. - pi / 2.)

##############################################################################
#
# TDist distribution - Standard student-t distribution
#
##############################################################################

immutable TDist <: ContinuousUnivariateDistribution
    df::Float64                         # non-integer degrees of freedom allowed
    TDist(d) = d > 0 ? new(float64(d)) : error("df must be positive")
end
@_jl_dist_1p TDist t
entropy(d::TDist) = ((d.df + 1.)/2.)*(digamma((d.df + 1.)/2.) - digamma((d.df)/2.)) + (1./2.)*log(d.df) + lbeta(d.df + 1., 1./2.)
mean(d::TDist) = d.df > 1 ? 0. : NaN
median(d::TDist) = 0.
modes(d::TDist) = [0.0]
var(d::TDist) = d.df > 2 ? d.df/(d.df-2) : d.df > 1 ? Inf : NaN
insupport(d::TDist, x::Number) = real_valued(x) && isfinite(x)
pdf(d::TDist, x::Real) = 1.0 / (sqrt(d.df) * beta(0.5, 0.5 * d.df)) * (1.0 + x^2 / d.df)^(-0.5*(d.df + 1.0))

##############################################################################
#
# univariate scaled, noncentral Student-t distribution with df degrees of freedom
# non-centrality parameter 'mean' and scale parameter 'sigma'
#
##############################################################################

immutable StDist <: ContinuousUnivariateDistribution
    df::Float64                         # non-integer degrees of freedom allowed
    mu::Float64
    sigma::Float64
    StDist(d,m,s) = d > 0 ? new(float64(d),float64(s),float64(s)) : error("df must be positive")
end
rand(d::StDist)= d.mu+d.sigma*ccall((:rt,Rmath),Float64,(Float64,),d.df)
mean(d::StDist) = d.df > 1 ? d.mu : NaN
median(d::StDist) = d.mu
modes(d::StDist) = [d.mu]
var(d::StDist) = d.df > 2 ? d.sigma^2*(d.df/(d.df-2)) : d.df > 1 ? Inf : NaN
insupport(d::StDist, x::Number) = real_valued(x) && isfinite(x)
pdf(d::StDist, x::Real) = 1.0 / (sqrt(d.df*sigma) * beta(0.5, 0.5 * d.df)) * (1.0 + (x-d.mu)^2 / (d.df*d.sigma^2))^(-0.5*(d.df + 1.0))

##############################################################################
#
# Multivariate scaled, noncentral Student-t distribution with df degrees of freedom
# non-centrality parameter 'mean' and parameterized in terms of the Choleski decomposed scale matrix 'covchol',
# (although a scale matrix can be entered to define the distribution)
#
##############################################################################

immutable MStDist <: ContinuousMultivariateDistribution
  df::Float64
  mean::Vector{Float64}
  covchol::Cholesky{Float64}
  function MStDist(d, m, c)
    if d<=0
	error("df must be positive")
    elseif length(m) == size(c, 1) == size(c, 2)
      new(d, m, c)
    else
      error("Dimensions of mean vector and scale matrix do not match")
    end
  end
end

MStDist(df::Float64,mean::Vector{Float64}, cov::Matrix{Float64}) = MStDist(df,mean,cholfact(cov))
MStDist(df::Float64,mean::Vector{Float64}) = MStDist(df,mean, eye(length(mean)))
MStDist(df::Float64,cov::Matrix{Float64}) = MStDist(df,zeros(size(cov, 1)), cov)
MStDist(df::Float64) = MStDist(df,zeros(2), eye(2))

mean(d::MStDist) = d.df > 1 ? d.mean : NaN
var(d::MStDist) = d.df > 2 ? (U = d.covchol[:U]; U'U*(d.df/(d.df-2))) : NaN

function rand(d::MStDist)
  z = randn(length(d.mean))
  x = rand(Chisq(d.df))
  return d.mean + d.covchol[:U]'z*sqrt(d.df/x)
end

function rand!(d::MStDist, X::Matrix)
  k = length(mean(d))
  m, n = size(X)
  if m == k return d.covchol[:U]'randn!(X)*sqrt(d.df/rand(Chisq(d.df),n))'[ones(Int,m),:] + d.mean[:,ones(Int,n)] end
  if n == k return randn!(X) * d.covchol[:U]*sqrt(d.df/rand(Chisq(d.df),m))[:,ones(Int,n)] + d.mean'[ones(Int,m),:] end
  error("Wrong dimensions")
end

function logpdf{T <: Real}(d::MStDist, x::Vector{T})
  k = length(d.mean)
  u = x - d.mean
  z = d.covchol \ u  # This is equivalent to inv(cov) * u, but much faster
  return lgamma((d.df+k)/2) - (lgamma(d.df/2) + (k/2)*(log(d.df) + log(pi))) - sum(log(diag(d.covchol[:U]))) -(d.df+k)/2*log(1+(1/d.df)*dot(u,z))
end

pdf{T <: Real}(d::MStDist, x::Vector{T}) = exp(logpdf(d, x))

##############################################################################
#
# Symmetric triangular distribution from Distributions Handbook
#
# TODO: Rearrange methods
#
##############################################################################

immutable Triangular <: ContinuousUnivariateDistribution
  location::Float64
  scale::Float64
  function Triangular(l, s)
    if s > 0.0
      new(float64(l), float64(s))
    else
      error("scale must be positive")
    end
  end
end
Triangular(location::Real) = Triangular(location, 1.0)
Triangular() = Triangular(0.0, 1.0)

function insupport(d::Triangular, x::Number)
  o = real_valued(x) && isfinite(x)
  return o && d.location - d.scale <= x <= d.location + d.scale
end
kurtosis(d::Triangular) = d.scale^4 / 15.0
mean(d::Triangular) = d.location
median(d::Triangular) = d.location
modes(d::Triangular) = [d.location]
function pdf(d::Triangular, x::Real)
  if insupport(d, x)
    return -abs(x - d.location) / (d.scale^2) + 1.0 / d.scale
  else
    return 0.0
  end
end
function rand(d::Triangular)
  xi1, xi2 = rand(), rand()
  return d.location + (xi1 - xi2) * d.scale
end
skewness(d::Triangular) = 0.0
var(d::Triangular) = d.scale^2 / 6.0

##############################################################################
#
# Uniform distribution
#
# TODO: Rearrange methods
#
##############################################################################

immutable Uniform <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
    Uniform(a, b) = a < b ? new(float64(a), float64(b)) : error("a < b required for range [a, b]")
end
Uniform() = Uniform(0, 1)
@_jl_dist_2p Uniform unif
entropy(d::Uniform) = log(d.b - d.a + 1.)
mean(d::Uniform) = (d.a + d.b) / 2.
median(d::Uniform) = (d.a + d.b)/2.
modes(d::Uniform) = error("The uniform distribution has no modes")
rand(d::Uniform) = d.a + (d.b - d.a) * rand()
var(d::Uniform) = (w = d.b - d.a; w * w / 12.)
insupport(d::Uniform, x::Number) = real_valued(x) && d.a <= x <= d.b
skewness(d::Uniform) = 0.0
kurtosis(d::Uniform) = -6.0 / 5.0

immutable Weibull <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    Weibull(sh,sc) = 0 < sh && 0 < sc ? new(float64(sh), float64(sc)) : error("Both shape and scale must be positive")
end
Weibull(sh) = Weibull(sh, 1)
@_jl_dist_2p Weibull weibull
mean(d::Weibull) = d.scale * gamma(1 + 1/d.shape)
var(d::Weibull) = d.scale^2*gamma(1 + 2/d.shape) - mean(d)^2
cdf(d::Weibull, x::Real) = 0 < x ? 1. - exp(-((x/d.scale)^d.shape)) : 0.
insupport(d::Weibull, x::Number) = real_valued(x) && isfinite(x) && 0 <= x

##
##
## Multinomial distribution
##
##

immutable Multinomial <: DiscreteMultivariateDistribution
  n::Int
  prob::Vector{Float64}
  function Multinomial{T <: Real}(n::Integer, p::Vector{T})
    if n <= 0
      error("Multinomial: n must be positive")
    end
    sump = 0.
    for i in 1:length(p)
      if p[i] < 0.
        error("Multinomial: probabilities must be non-negative")
      end
      sump += p[i]
    end
    new(int(n), p ./ sump)
  end
end

function Multinomial(n::Integer, d::Integer)
  if d <= 1
    error("d must be greater than 1")
  end
  Multinomial(n, ones(d) / d)
end

Multinomial(d::Integer) = Multinomial(1, d)

mean(d::Multinomial) = d.n .* d.prob
var(d::Multinomial)  = d.n .* d.prob .* (1 - d.prob)

function insupport{T <: Real}(d::Multinomial, x::Vector{T})
  n = length(x)
  if length(d.prob) != n
    return false
  end
  s = 0.0
  for i in 1:n
    if x[i] < 0. || !integer_valued(x[i])
      return false
    end
    s += x[i]
  end
  if abs(s - d.n) > 10e-8
    return false
  end
  return true
end

function logpdf{T <: Real}(d::Multinomial, x::Vector{T})
  !insupport(d, x) ? -Inf : lgamma(d.n + 1) - sum(lgamma(x + 1)) + sum(x .* log(d.prob))
end

pdf{T <: Real}(d::Multinomial, x::Vector{T}) = exp(logpdf(d, x))

function rand(d::Multinomial)
  n = d.n
  l = length(d.prob)
  s = zeros(Int, l)
  psum = 1.0
  for j = 1:(l - 1)
    s[j] = int(ccall((:rbinom, Rmath), Float64, (Float64, Float64), n, d.prob[j] / psum))
    n -= s[j]
    if n == 0
      break
    end
    psum -= d.prob[j]
  end
  s[end] = n
  s
end

##
##
## Dirichlet distribution
##
##

immutable Dirichlet <: ContinuousMultivariateDistribution
  alpha::Vector{Float64}
  function Dirichlet{T <: Real}(alpha::Vector{T})
    for el in alpha
      if el < 0. error("Dirichlet: elements of alpha must be non-negative") end
    end
    new(float64(alpha))
  end
end

Dirichlet(dim::Integer) = Dirichlet(ones(dim))

mean(d::Dirichlet) = d.alpha ./ sum(d.alpha)

function var(d::Dirichlet)
  alpha0 = sum(d.alpha)
  d.alpha .* (alpha0 - d.alpha) / (alpha0^2 * (alpha0 + 1))
end

function insupport{T <: Real}(d::Dirichlet, x::Vector{T})
  n = length(x)
  if length(d.alpha) != n
    return false
  end
  s = 0.0
  for i in 1:n
    if x[i] < 0.
      return false
    end
    s += x[i]
  end
  if abs(s - 1.) > 10e-8
    return false
  end
  return true
end

function pdf{T <: Real}(d::Dirichlet, x::Vector{T})
  if !insupport(d, x)
    error("x not in the support of Dirichlet distribution")
  end
  b = prod(gamma(d.alpha)) / gamma(sum(d.alpha))
  (1 / b) * prod(x.^(d.alpha - 1))
end

function logpdf{T <: Real}(d::Dirichlet, x::Vector{T})
  if !insupport(d, x)
    error("x not in the support of Dirichlet distribution")
  end
  b = sum(lgamma(d.alpha)) - lgamma(sum(d.alpha))
  dot((d.alpha - 1), log(x)) - b
end

function rand(d::Dirichlet)
  x = [rand(Gamma(el)) for el in d.alpha]
  x ./ sum(x)
end

function rand!(d::Dirichlet, X::Matrix)
  m, n = size(X)
  for i in 1:n
    X[:,i] = rand(Gamma(d.alpha[i]), m)
  end
  for i in 1:m
    isum = sum(X[i,:])
    for j in 1:n
      X[i,j] /= isum
    end
  end
  return X
end


##
##
## Categorical distribution
##
##

immutable Categorical <: DiscreteUnivariateDistribution
  prob::Vector{Float64}
  function Categorical{T <: Real}(p::Vector{T})
    if length(p) <= 1
      error("Categorical: there must be at least two categories")
    end
    sump = 0.
    for i in 1:length(p)
      if p[i] < 0.
        error("Categorical: probabilities must be non-negative")
      end
      sump += p[i]
    end
    new(p ./ sump)
  end
end

function Categorical(d::Integer)
  if d <= 1
    error("d must be greater than 1")
  end
  Categorical(ones(d) / d)
end

function insupport(d::Categorical, x::Real)
  integer_valued(x) && 1 <= x <= length(d.prob) && d.prob[x] != 0.0
end

pdf(d::Categorical, x::Real) = !insupport(d, x) ? 0. : d.prob[x]

function rand(d::Categorical)
  l = length(d.prob)
  r = rand()
  for j = 1:l
    r -= d.prob[j]
    if r <= 0.0
      return j
    end
  end
  return l
end

##
##
## Sample from arbitrary arrays
##
##

function sample{T <: Real}(a::AbstractArray, probs::Vector{T})
  i = rand(Categorical(probs))
  a[i]
end

function sample(a::AbstractArray)
  n = length(a)
  probs = ones(n) ./ n
  sample(a, probs)
end

const minfloat = realmin(Float64)
const oneMeps  = 1. - eps()
const llmaxabs = log(-log(minfloat))
const logeps   = log(eps())
abstract Link                           # Link types define linkfun, linkinv, mueta,
                                        # valideta and validmu.

chkpositive(x::Real) = isfinite(x) && 0. < x ? x : error("argument must be positive")
chkfinite(x::Real) = isfinite(x) ? x : error("argument must be finite")
clamp01(x::Real) = clamp(x, minfloat, oneMeps)
chk01(x::Real) = 0. < x < 1. ? x : error("argument must be in (0,1)")

type CauchitLink  <: Link end
linkfun (l::CauchitLink,   mu::Real) = tan(pi * (mu - 0.5))
linkinv (l::CauchitLink,  eta::Real) = 0.5 + atan(eta) / pi
mueta   (l::CauchitLink,  eta::Real) = 1. /(pi * (1 + eta * eta))
valideta(l::CauchitLink,  eta::Real) = chkfinite(eta)
validmu (l::CauchitLink,   mu::Real) = chk01(mu)

type CloglogLink  <: Link end
linkfun (l::CloglogLink,   mu::Real) = log(-log(1. - mu))
linkinv (l::CloglogLink,  eta::Real) = -expm1(-exp(eta))
mueta   (l::CloglogLink,  eta::Real) = exp(eta) * exp(-exp(eta))
valideta(l::CloglogLink,  eta::Real) = abs(eta) < llmaxabs? eta: error("require abs(eta) < $llmaxabs")
validmu (l::CloglogLink,   mu::Real) = chk01(mu)

type IdentityLink <: Link end
linkfun (l::IdentityLink,  mu::Real) = mu
linkinv (l::IdentityLink, eta::Real) = eta
mueta   (l::IdentityLink, eta::Real) = 1.
valideta(l::IdentityLink, eta::Real) = chkfinite(eta)
validmu (l::IdentityLink,  mu::Real) = chkfinite(mu)

type InverseLink  <: Link end
linkfun (l::InverseLink,   mu::Real) =  1. / mu
linkinv (l::InverseLink,  eta::Real) =  1. / eta
mueta   (l::InverseLink,  eta::Real) = -1. / (eta * eta)
valideta(l::InverseLink,  eta::Real) = chkpositive(eta)
validmu (l::InverseLink,  eta::Real) = chkpositive(mu)

type LogitLink    <: Link end
linkfun (l::LogitLink,     mu::Real) = log(mu / (1 - mu))
linkinv (l::LogitLink,    eta::Real) = 1. / (1. + exp(-eta))
mueta   (l::LogitLink,    eta::Real) = (e = exp(-abs(eta)); f = 1. + e; e / (f * f))
valideta(l::LogitLink,    eta::Real) = chkfinite(eta)
validmu (l::LogitLink,     mu::Real) = chk01(mu)

type LogLink      <: Link end
linkfun (l::LogLink,       mu::Real) = log(mu)
linkinv (l::LogLink,      eta::Real) = exp(eta)
mueta   (l::LogLink,      eta::Real) = eta < logeps ? eps() : exp(eta)
valideta(l::LogLink,      eta::Real) = chkfinite(eta)
validmu (l::LogLink,       mu::Real) = chkpositive(mu)

type ProbitLink   <: Link end
linkfun (l::ProbitLink,    mu::Real) = ccall((:qnorm5, Rmath), Float64,
                                             (Float64,Float64,Float64,Int32,Int32),
                                             mu, 0., 1., 1, 0)
linkinv (l::ProbitLink,   eta::Real) = (1. + erf(eta/sqrt(2.))) / 2.
mueta   (l::ProbitLink,   eta::Real) = exp(-0.5eta^2) / sqrt(2.pi)
valideta(l::ProbitLink,   eta::Real) = chkfinite(eta)
validmu (l::ProbitLink,    mu::Real) = chk01(mu)
                                        # Vectorized methods, including validity checks
function linkfun{T<:Real}(l::Link, mu::AbstractArray{T,1})
    [linkfun(l, validmu(l, m)) for m in mu]
end

function linkinv{T<:Real}(l::Link, eta::AbstractArray{T,1})
    [linkinv(l, valideta(l, et)) for et in eta]
end

function mueta{T<:Real}(l::Link, eta::AbstractArray{T,1})
    [mueta(l, valideta(l, et)) for et in eta]
end

canonicallink(d::Gamma)     = InverseLink()
canonicallink(d::Normal)    = IdentityLink()
canonicallink(d::Bernoulli) = LogitLink()
canonicallink(d::Poisson)   = LogLink()

include("show.jl")

include("fit.jl")

end  #module
