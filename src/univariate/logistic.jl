immutable Logistic <: ContinuousUnivariateDistribution
    location::Real
    scale::Real
    function Logistic(l::Real, s::Real)
    	s > zero(s) || error("scale must be positive")
    	new(float64(l), float64(s))
    end
    Logistic(l::Real) = new(float64(l), 1.0)
    Logistic() = new(0.0, 1.0)
end

## Support
@continuous_distr_support Logistic -Inf Inf

## Properties
mean(d::Logistic) = d.location
median(d::Logistic) = d.location
mode(d::Logistic) = d.location

std(d::Logistic) = pi * d.scale / sqrt(3.0)
var(d::Logistic) = (pi * d.scale)^2 / 3.0

skewness(d::Logistic) = 0.0
kurtosis(d::Logistic) = 1.2

entropy(d::Logistic) = log(d.scale) + 2.0

## Functions
function pdf(d::Logistic, x::Real)
    a = exp(-abs((x-d.location)/d.scale))
    a / (d.scale * (1+a)^2)
end
function logpdf(d::Logistic, x::Real)
    u = -abs((x-d.location)/d.scale)
    u - 2*log1pexp(u) - log(d.scale)
end

cdf(d::Logistic, x::Real) = logistic((x-d.location)/d.scale)
ccdf(d::Logistic, x::Real) = logistic((d.location-x)/d.scale)
logcdf(d::Logistic, x::Real) = -log1pexp((d.location-x)/d.scale)
logccdf(d::Logistic, x::Real) = -log1pexp((x-d.location)/d.scale)

quantile(d::Logistic, p::Real) = @checkquantile p d.location + d.scale*logit(p)
cquantile(d::Logistic, p::Real) = @checkquantile p d.location - d.scale*logit(p)
invlogcdf(d::Logistic, lp::Real) = d.location - d.scale*logexpm1(-lp)
invlogccdf(d::Logistic, lp::Real) = d.location + d.scale*logexpm1(-lp)


function gradlogpdf(d::Logistic, x::Real)
  expterm = exp((d.location - x) / d.scale)
  ((2 * expterm) / (1 + expterm) - 1) / d.scale
end

mgf(d::Logistic, t::Real) = exp(t*d.location)/sinc(d.scale*t)

function cf(d::Logistic, t::Real)
    a = (pi*t)*d.scale
    a == zero(a) ? complex(one(a)) : exp(im*t*d.location) * a / sinh(a)    
end
