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

@continuous_distr_support Logistic -Inf Inf


mean(d::Logistic) = d.location
median(d::Logistic) = d.location
mode(d::Logistic) = d.location
modes(d::Logistic) = [d.location]

std(d::Logistic) = pi * d.scale / sqrt(3.0)
var(d::Logistic) = (pi * d.scale)^2 / 3.0

skewness(d::Logistic) = 0.0
kurtosis(d::Logistic) = 1.2

entropy(d::Logistic) = log(d.scale) + 2.0

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

quantile(d::Logistic, p::Real) = d.location + d.scale*logit(p)
cquantile(d::Logistic, p::Real) = d.location - d.scale*logit(p)
invlogcdf(d::Logistic, lp::Real) = d.location - d.scale*logexpm1(-lp)
invlogccdf(d::Logistic, lp::Real) = d.location + d.scale*logexpm1(-lp)

rand(d::Logistic) = quantile(d, rand())


function mgf(d::Logistic, t::Real)
    m, b = d.location, d.scale
    exp(t * m) * (pi * b * t) / sin(pi * b * t)
end

function cf(d::Logistic, t::Real)
    m, b = d.location, d.scale
    exp(im * t * m) * (pi * b * t) / sinh(pi * b * t)
end
