# Fréchet Distribution

immutable Frechet <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    function Frechet(sh::Real, sc::Real)
    	zero(sh) < sh && zero(sc) < sc || error("Both shape and scale must be positive")
    	new(float64(sh), float64(sc))
    end
end

Frechet(sh::Real) = Frechet(sh, 1.0)

## Support
@continuous_distr_support Frechet 0.0 Inf

## Properties
mean(d::Frechet) = d.shape > 1.0 ? d.scale * gamma(1.0 - 1.0 / d.shape) : Inf
median(d::Frechet) = d.scale * log(2)^(-1.0 / d.shape)

mode(d::Frechet) = (ik = -1.0/d.shape; d.scale * (1.0-ik)^ik)

var(d::Frechet) = d.shape > 2.0 ? d.scale^2 * gamma(1.0 - 2.0 / d.shape) - mean(d)^2 : NaN

function skewness(d::Frechet)
    d.shape <= 3.0 && return NaN
    tmp_mean = mean(d)
    tmp_var = var(d)
    tmp = gamma(1.0 - 3.0 / d.shape) * d.scale^3
    tmp -= 3.0 * tmp_mean * tmp_var
    tmp -= tmp_mean^3
    return tmp / tmp_var / sqrt(tmp_var)
end

function kurtosis(d::Frechet)
    d.shape <= 4.0 && return NaN
    λ, k = d.scale, d.shape
    μ = mean(d)
    σ = std(d)
    γ = skewness(d)
    den = λ^4 * gamma(1.0 - 4.0 / k) -
          4.0 * γ * σ^3 * μ -
          6.0 * μ^2 * σ^2 - μ^4
    num = σ^4
    return den / num - 3.0
end

function entropy(d::Frechet)
    const γ = 0.57721566490153286060
    1.0 + γ / d.shape + γ + log(d.scale / d.shape)
end

## Functions
function pdf(d::Frechet, x::Real)
    x < zero(x) && return 0.0
    a = d.scale/x
    d.shape/d.scale * a^(d.shape+1.0) * exp(-a^d.shape)
end
function logpdf(d::Frechet, x::Real)
    x < zero(x) && return -Inf
    a = d.scale/x
    log(d.shape/d.scale) + (d.shape+1.0)*log(a) - a^d.shape
end

cdf(d::Frechet, x::Real) = x <= zero(x) ? 0.0 : exp(-((d.scale / x)^d.shape))
ccdf(d::Frechet, x::Real) = x <= zero(x) ? 1.0 : -expm1(-((d.scale / x)^d.shape))
logcdf(d::Frechet, x::Real) = x <= zero(x) ? -Inf : -(d.scale / x)^d.shape
logccdf(d::Frechet, x::Real) = x <= zero(x) ? 0.0 :  log1mexp(-((d.scale / x)^d.shape))

quantile(d::Frechet, p::Real) = @checkquantile p d.scale*(-log(p))^(-1/d.shape)
cquantile(d::Frechet, p::Real) = @checkquantile p d.scale*(-log1p(-p))^(-1/d.shape)
invlogcdf(d::Frechet, lp::Real) = lp > zero(lp) ? NaN : d.scale*(-lp)^(-1/d.shape)
invlogccdf(d::Frechet, lp::Real) = lp > zero(lp) ? NaN : d.scale*(-log1mexp(lp))^(-1/d.shape)


function gradlogpdf(d::Frechet, x::Real)
  insupport(Frechet, x) ? -(d.shape + 1.0) / x + d.shape * (d.scale^d.shape) * x^(-d.shape - 1.0)  : 0.0
end

## Sampling
rand(d::Frechet) = d.scale*Base.Random.randmtzig_exprnd()^(-1/d.shape)
