immutable Laplace <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Laplace(l::Real, s::Real)
        s > zero(s) || error("scale must be positive")
        new(float64(l), float64(s))
    end
end

Laplace(location::Real) = Laplace(location, 1.0)
Laplace() = Laplace(0.0, 1.0)

typealias Biexponential Laplace

## Support
@continuous_distr_support Laplace -Inf Inf

## Properties
mean(d::Laplace) = d.location
median(d::Laplace) = d.location
mode(d::Laplace) = d.location

std(d::Laplace) = sqrt2 * d.scale
var(d::Laplace) = 2.0 * d.scale^2
skewness(d::Laplace) = 0.0
kurtosis(d::Laplace) = 3.0

entropy(d::Laplace) = log(2.0 * d.scale) + 1.0

## Functions
pdf(d::Laplace, x::Real) = 0.5exp(-abs(x - d.location)/d.scale) / d.scale
logpdf(d::Laplace, x::Real) = -log(2.0 * d.scale) - abs(x - d.location) / d.scale

cdf(d::Laplace, x::Real) = x < d.location ? 0.5*exp((x-d.location)/d.scale) : 0.5*(2.0-exp((d.location-x)/d.scale))
ccdf(d::Laplace, x::Real) = x < d.location ? 0.5-0.5*expm1((x-d.location)/d.scale) : 0.5*exp((d.location-x)/d.scale)
logcdf(d::Laplace, x::Real) = x < d.location ? loghalf + ((x-d.location)/d.scale) : loghalf + log2mexp((d.location-x)/d.scale)
logccdf(d::Laplace, x::Real) = x < d.location ? loghalf + log2mexp((x-d.location)/d.scale) : loghalf + ((d.location-x)/d.scale)

quantile(d::Laplace, p::Real) = p < 0.5 ? d.location + d.scale*log(2.0*p) : d.location - d.scale*log(2.0*(1.0-p))
cquantile(d::Laplace, p::Real) = p >= 0.5 ? d.location + d.scale*log(2.0*(1.0-p)) : d.location - d.scale*log(2.0*p)
invlogcdf(d::Laplace, lp::Real) = lp < loghalf ? d.location + d.scale*(logtwo + lp) : d.location - d.scale*(logtwo + log1mexp(lp))
invlogccdf(d::Laplace, lp::Real) = lp >= loghalf ? d.location + d.scale*(logtwo + log1mexp(lp)) : d.location - d.scale*(logtwo + lp)

function mgf(d::Laplace, t::Real)
    st = d.scale*t
    exp(t * d.location) / ((1.0-st)*(1.0+st))
end
function cf(d::Laplace, t::Real)
    st = d.scale*t
    exp(im * t * d.location) / ((1.0-st)*(1.0+st))
end

function gradlogpdf(d::Laplace, x::Real)
    d.location != x || error("Gradient is undefined at the location point")
    x > d.location ? - 1.0 / d.scale : 1.0 / d.scale
end


## Sampling
function rand(d::Laplace) 
    er = Base.Random.randmtzig_exprnd() 
    randbool() ? d.location - d.scale * er : d.location + d.scale * er
end

## Fitting
function fit_mle(::Type{Laplace}, x::Array)
    a = median(x)
    Laplace(a, mad(x, a))
end
