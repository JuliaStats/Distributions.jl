immutable Weibull <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    function Weibull(sh::Real, sc::Real)
    	zero(sh) < sh && zero(sc) < sc || error("Both shape and scale must be positive")
    	new(float64(sh), float64(sc))
    end
end

Weibull(sh::Real) = Weibull(sh, 1.0)

## Support
@continuous_distr_support Weibull 0.0 Inf

## Properties
mean(d::Weibull) = d.scale * gamma(1.0 + 1.0 / d.shape)
median(d::Weibull) = d.scale * log(2.0)^(1.0 / d.shape)

mode(d::Weibull) = d.shape > 1.0 ? (ik = 1.0/d.shape; d.scale * (1.0-ik)^ik) : 0.0

var(d::Weibull) = d.scale^2 * gamma(1.0 + 2.0 / d.shape) - mean(d)^2

function skewness(d::Weibull)
    tmp = gamma(1.0 + 3.0 / d.shape) * d.scale^3
    tmp -= 3.0 * mean(d) * var(d)
    tmp -= mean(d)^3
    return tmp / std(d)^3
end

function kurtosis(d::Weibull)
    λ, k = d.scale, d.shape
    μ = mean(d)
    σ = std(d)
    γ = skewness(d)
    den = λ^4 * gamma(1.0 + 4.0 / k) -
          4.0 * γ * σ^3 * μ -
          6.0 * μ^2 * σ^2 - μ^4
    num = σ^4
    return den / num - 3.0
end

function entropy(d::Weibull)
    λ, k = d.scale, d.shape
    return ((k - 1.0) / k) * -digamma(1.0) + log(λ / k) + 1.0
end

## Functions
function pdf(d::Weibull, x::Real)
    x < zero(x) && return 0.0
    a = x/d.scale
    d.shape/d.scale * a^(d.shape-1.0) * exp(-a^d.shape)
end
function logpdf(d::Weibull, x::Real)
    x < zero(x) && return -Inf
    a = x/d.scale
    log(d.shape/d.scale) + (d.shape-1.0)*log(a) - a^d.shape
end

cdf(d::Weibull, x::Real) = x <= zero(x) ? 0.0 : -expm1(-((x / d.scale)^d.shape))
ccdf(d::Weibull, x::Real) = x <= zero(x) ? 1.0 : exp(-((x / d.scale)^d.shape))
logcdf(d::Weibull, x::Real) = x <= zero(x) ? -Inf : log1mexp(-((x / d.scale)^d.shape))
logccdf(d::Weibull, x::Real) = x <= zero(x) ? 0.0 : -(x / d.scale)^d.shape

quantile(d::Weibull, p::Real) = @checkquantile p d.scale*(-log1p(-p))^(1/d.shape)
cquantile(d::Weibull, p::Real) = @checkquantile p d.scale*(-log(p))^(1/d.shape)
invlogcdf(d::Weibull, lp::Real) = lp > zero(lp) ? NaN : d.scale*(-log1mexp(lp))^(1/d.shape)
invlogccdf(d::Weibull, lp::Real) = lp > zero(lp) ? NaN : d.scale*(-lp)^(1/d.shape)


function gradloglik(d::Weibull, x::Float64)
  insupport(Weibull, x) ? (d.shape - 1.0) / x - d.shape * x^(d.shape - 1.0) / (d.scale^d.shape) : 0.0
end

## Sampling
rand(d::Weibull) = d.scale*Base.Random.randmtzig_exprnd()^(1/d.shape)
