# Eric Feltham (https://github.com/emfeltham)
# zero-inflated distributions.jl
# logpdf for ZeroInflatedPoisson written by Chris Fisher (https://github.com/itsdfish)

"""
    ZeroInflatedPoisson(λ, p)
A *Zero-Inflated Poisson distribution* is a mixture distribution in which data arise from two processes. The first process is is a Poisson distribution, with mean λ, that descibes the number of independent events occurring within a unit time interval. Zeros may arise from this process, or modeleled as a Bernoulli trial, with probability of observing an excess zero p. As p approaches 0, the distribution tends toward Poisson(λ).
```math
P(X = 0) = p + (1 - p) e^{-\\lambda}

P(X = k) = (1 - p) \\frac{\\lambda^k}{k!} e^{-\\lambda}, \\quad \\text{ for } k = 0,1,2,\\ldots.
```
```julia
ZeroInflatedPoisson()      # Zero-Inflated Poisson distribution with rate parameter 1, and probability of observing a zero 0.5
ZeroInflatedPoisson(λ)       # ZeroInflatedPoisson distribution with rate parameter λ, and probability of observing a zero 0.5
params(d)        # Get the parameters, i.e. (λ, w)
mean(d)          # Get the mean of the mixture distribution
var(d)           # Get the variance of the mixture distribution
```
External links:
* [Zero-inflated Poisson Regression on UCLA IDRE Statistical Consulting](https://stats.idre.ucla.edu/stata/dae/zero-inflated-poisson-regression/)
* [Zero-inflated model on Wikipedia](https://en.wikipedia.org/wiki/Zero-inflated_model)
* McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/9780429029608

"""

using Distributions, Random, SpecialFunctions, LambertW
using LogExpFunctions

# explicit imports to add methods
import Distributions: cdf, logpdf, rand, @check_args, suffstats, fit_mle
import Statistics: mean, var, quantile

struct ZeroInflatedPoisson{T<:Real} <: DiscreteUnivariateDistribution
  λ::T
  p::T

  function ZeroInflatedPoisson{T}(λ::T, p::T) where {T <: Real}
      return new{T}(λ, p)
    end
end

function ZeroInflatedPoisson(λ::T, p::T; check_args = true) where {T <: Real}
  if check_args
    @check_args(Poisson, λ >= zero(λ))
    @check_args(ZeroInflatedPoisson, zero(p) <= p <= one(p))
  end
  return ZeroInflatedPoisson{T}(λ, p)
end

ZeroInflatedPoisson(λ::Real, p::Real) = ZeroInflatedPoisson(promote(λ, p)...)
ZeroInflatedPoisson(λ::Integer, p::Integer) = ZeroInflatedPoisson(float(λ), float(w))
ZeroInflatedPoisson(λ::Real) = ZeroInflatedPoisson(λ, 0.5)
ZeroInflatedPoisson() = ZeroInflatedPoisson(1.0, 0.5, check_args = false)

### Statistics

mean(d::ZeroInflatedPoisson) = (1 - d.p) * d.λ # check

var(d::ZeroInflatedPoisson) = λ * log1p(-p) * log1p(p * λ)

#### Conversions

function convert(::Type{ZeroInflatedPoisson{T}}, λ::Real, p::Real) where {T<:Real}
  return ZeroInflatedPoisson(T(λ), T(p))
end

function convert(::Type{ZeroInflatedPoisson{T}}, d::ZeroInflatedPoisson{S}) where {T <: Real, S <: Real}
  return ZeroInflatedPoisson(T(d.λ), T(d.p), check_args = false)
end

#### Parameters

import StatsBase:params

params(d::ZeroInflatedPoisson) = (d.λ, d.p,)
partype(::ZeroInflatedPoisson{T}) where {T} = T

#### Evaluation

#= 
if ( y[i] == 0 ) target += log_mix(p, 0, poisson_lpmf(0 | lambda));
<=> if ( y[i] == 0 ) target += log( p + (1-p)*exp(-lambda) );
if ( y[i] > 0 ) target += log1m(p) + poisson_lpmf(y[i] | lambda); (Stan code from McElreath Ch. 12)
=#
function logpdf(d::ZeroInflatedPoisson, y::Int)
  LL = 0.0
  if y == 0
    LLs = zeros(typeof(log(d.λ)), 2)
    LLs[1] = log(d.p)
    LLs[2] = log1p(-d.p) - d.λ
    LL = logsumexp(LLs)
  else
    LL = log1p(-d.p) + logpdf(LogPoisson(log(d.λ)), y)
  end
  return LL
end

# cdf
# reference pzipois(q, lambda, pstr0 = 0) from VGAM R Pkg.
function cdf(d::ZeroInflatedPoisson, x::Int)
  pd = Poisson(d.λ)

  deflat_limit = -1.0 / expm1(d.λ)

  if x < 0
    out = 0
  elseif d.p < deflat_limit # is this condition already resolved by @check args?
    out = NaN
  else
    out = d.p + (1 - d.p) * cdf(pd, x)
  end
  return out
end

# quantile
# reference qzipois() from VGAM R Pkg.
function quantile(d::ZeroInflatedPoisson, q::Real)

  deflat_limit = -1.0 / expm1(d.λ)

  if (q <= d.p)
    out = 0
  elseif (d.p < deflat_limit)
    out = NaN
  elseif (d.p < q) & (deflat_limit <= d.p) & (q < 1)
    qp = (q - d.p) / (1 - d.p)
    pd = Poisson(d.λ)
    out = quantile(pd, qp)
  elseif p == 1
    out = Inf
  end
  return out
end

## Sampling

function rand(rng::AbstractRNG, d::ZeroInflatedPoisson)
  zo = rand(rng, Uniform(0, 1))
  return quantile(d, zo)
end

#### Fitting

struct ZeroInflatedPoissonStats <: SufficientStats
  sx::Float64   # (weighted) sum of x
  p0::Float64   # observed proportion of zeros
  tw::Float64   # total sample weight
end

suffstats(::Type{<:ZeroInflatedPoisson}, x::AbstractArray{T}) where {T<:Integer} = ZeroInflatedPoissonStats(
    sum(x),
    sum(x .== 0) / length(x),
    length(x)
  )

# weighted
function suffstats(::Type{<:ZeroInflatedPoisson}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    n = length(x)
    n == length(w) || throw(DimensionMismatch("Inconsistent array lengths."))
    sx = 0.
    tw = 0.
    p0 = 0.
    for i = 1 : n
        @inbounds wi = w[i]
        @inbounds sx += x[i] * wi
        tw += wi
        @inbounds p0i = (x[i] == 0) * wi
        p0 += p0i
    end
    return ZeroInflatedPoissonStats(sx, p0, tw)
end

function fit_mle(::Type{<:ZeroInflatedPoisson}, ss::ZeroInflatedPoissonStats)
  m = ss.sx / ss.tw
  s = m / (1 - ss.p0)

  λhat = lambertw(-s * exp(-s)) + s
  phat = 1 - (m / λhat)

  return ZeroInflatedPoisson(λhat, phat)
end

# write a method that allows fit_mle(dtype, x), fit_mle(dtype, x, w) for ZeroInflatedPoisson AND Poisson (it is missing -- it does exist for Normal)
function fit_mle(::Type{<:ZeroInflatedPoisson}, x::AbstractArray{T}) where T<:Real
  pstat = suffstats(ZeroInflatedPoisson, x)
  return fit_mle(ZeroInflatedPoisson, pstat)
end

function fit_mle(::Type{<:ZeroInflatedPoisson}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
  pstat = suffstats(ZeroInflatedPoisson, x, w)
  return fit_mle(ZeroInflatedPoisson, pstat)
end
