"""
    ZeroInflatedPoisson(λ, p)
A *Zero-Inflated Poisson distribution* is a mixture distribution in which data arise from two processes. The first process is is a Poisson distribution, with mean λ, that descibes the number of independent events occurring within a unit time interval:
```math
P(X = k) = (1 - p) \\frac{\\lambda^k}{k!} e^{-\\lambda}, \\quad \\text{ for } k = 0,1,2,\\ldots.
```
Zeros may arise from this process, an additional Bernoulli process, where the probability of observing an excess zero is given as p:
```math
P(X = 0) = p + (1 - p) e^{-\\lambda}
```
As p approaches 0, the distribution tends toward Poisson(λ).
```julia
ZeroInflatedPoisson()        # Zero-Inflated Poisson distribution with rate parameter 1, and probability of observing a zero 0.5
ZeroInflatedPoisson(λ)       # ZeroInflatedPoisson distribution with rate parameter λ, and probability of observing a zero 0.5
params(d)                    # Get the parameters, i.e. (λ, w)
mean(d)                      # Get the mean of the mixture distribution
var(d)                       # Get the variance of the mixture distribution
```
External links:
* [Zero-inflated Poisson Regression on UCLA IDRE Statistical Consulting](https://stats.idre.ucla.edu/stata/dae/zero-inflated-poisson-regression/)
* [Zero-inflated model on Wikipedia](https://en.wikipedia.org/wiki/Zero-inflated_model)
* McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/9780429029608

"""
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
ZeroInflatedPoisson(λ::Integer, p::Integer) = ZeroInflatedPoisson(float(λ), float(p))
ZeroInflatedPoisson(λ::Real) = ZeroInflatedPoisson(λ, 0.5)
ZeroInflatedPoisson() = ZeroInflatedPoisson(1.0, 0.5, check_args = false)

### Statistics

mean(d::ZeroInflatedPoisson) = (1 - d.p) * d.λ

var(d::ZeroInflatedPoisson) = d.λ * log1p(-d.p) * log1p(d.p * d.λ)

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

function logpdf(d::ZeroInflatedPoisson, y::Int)
  LL = 0.0
  if y == 0
    LLs = zeros(typeof(log(d.λ)), 2)
    LLs[1] = log(d.p)
    LLs[2] = log1p(-d.p) - d.λ
    LL = logsumexp(LLs)
  else
    LL = log1p(-d.p) + logpdf(Poisson(d.λ), y)
  end
  return LL
end

function cdf(d::ZeroInflatedPoisson, x::Int)
  pd = Poisson(d.λ)

  deflat_limit = -1.0 / expm1(d.λ)

  if x < 0
    out = 0.0
  elseif d.p < deflat_limit
    out = NaN
  else
    out = d.p + (1 - d.p) * cdf(pd, x)
  end
  return out
end

# quantile
function quantile(d::ZeroInflatedPoisson, q::Real)

  deflat_limit = -1.0 / expm1(d.λ)

  if (q <= d.p)
    out = 0
  elseif (d.p < deflat_limit)
    out = convert(Int64, NaN)
  elseif (d.p < q) & (deflat_limit <= d.p) & (q < 1.0)
    qp = (q - d.p) / (1.0 - d.p)
    pd = Poisson(d.λ)
    out = quantile(pd, qp) # handles d.p == 1 as InexactError(Inf)
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

function fit_mle(::Type{<:ZeroInflatedPoisson}, x::AbstractArray{T}) where T<:Real
  pstat = suffstats(ZeroInflatedPoisson, x)
  return fit_mle(ZeroInflatedPoisson, pstat)
end

function fit_mle(::Type{<:ZeroInflatedPoisson}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
  pstat = suffstats(ZeroInflatedPoisson, x, w)
  return fit_mle(ZeroInflatedPoisson, pstat)
end
