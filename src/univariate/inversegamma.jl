##############################################################################
#
# InverseGamma distribution, cf Bayesian Theory (Barnardo & Smith) p 119
#
#  Note that B&S parametrize in terms of shape/rate, but this is uses
#  shape/scale so that it is consistent with the implementation of Gamma
#
##############################################################################

immutable InverseGamma <: ContinuousUnivariateDistribution
    shape::Float64 # location
    scale::Float64 # scale
    function InverseGamma(sh::Real, sc::Real)
    	sh > zero(sh) && sc > zero(sc) || error("Both shape and scale must be positive")
    	new(float64(sh), float64(sc))
    end
end

scale(d::InverseGamma) = d.scale
rate(d::InverseGamma) = 1.0 / d.scale

insupport(::InverseGamma, x::Real) = zero(x) <= x < Inf
insupport(::Type{InverseGamma}, x::Real) = zero(x) <= x < Inf


mean(d::InverseGamma) = d.shape > 1.0 ? (1.0 / d.scale) / (d.shape - 1.0) : Inf
var(d::InverseGamma) = d.shape > 2.0 ? (1.0 / d.scale)^2 / ((d.shape - 1.0)^2 * (d.shape - 2.0)) : Inf
skewness(d::InverseGamma) = d.shape > 3.0 ? (4.0 * sqrt(d.shape - 2.0)) / (d.shape - 3.0) : NaN
kurtosis(d::InverseGamma) = d.shape > 4.0 ? (30.0 * d.shape - 66.0) / ((d.shape - 3.0) * (d.shape - 4.0)) : NaN

mode(d::InverseGamma) = (1.0 / d.scale) / (d.shape + 1.0)
modes(d::InverseGamma) = [mode(d)]

cdf(d::InverseGamma, x::Real) = ccdf(Gamma(d.shape, d.scale), 1.0 / x)
ccdf(d::InverseGamma, x::Real) = cdf(Gamma(d.shape, d.scale), 1.0 / x)
logcdf(d::InverseGamma, x::Real) = logccdf(Gamma(d.shape, d.scale), 1.0 / x)
logccdf(d::InverseGamma, x::Real) = logcdf(Gamma(d.shape, d.scale), 1.0 / x)

quantile(d::InverseGamma, p::Real) = 1.0 / cquantile(Gamma(d.shape, d.scale), p)
cquantile(d::InverseGamma, p::Real) = 1.0 / quantile(Gamma(d.shape, d.scale), p)
invlogcdf(d::InverseGamma, p::Real) = 1.0 / invlogccdf(Gamma(d.shape, d.scale), p)
invlogccdf(d::InverseGamma, p::Real) = 1.0 / invlogcdf(Gamma(d.shape, d.scale), p)

function entropy(d::InverseGamma)
    a, b = d.shape, d.scale
    a - log(b) + lgamma(a) - (1.0 + a) * digamma(a)
end


function mgf(d::InverseGamma, t::Real)
    a, b = d.shape, d.scale
    (2 * (-b * t)^(a / 2)) / gamma(a) * besselk(a, sqrt(-4.0 * b * t))
end

function cf(d::InverseGamma, t::Real)
    a, b = d.shape, d.scale
    (2 * (-im * b * t)^(a / 2)) / gamma(a) * besselk(a, sqrt(-4.0 * im * b * t))
end

pdf(d::InverseGamma, x::Real) = exp(logpdf(d, x))

function logpdf(d::InverseGamma, x::Real)
    a, b = d.shape, d.scale
    -(a * log(b)) - lgamma(a) - ((a + 1.0) * log(x)) - 1.0/ (b * x)
end

rand(d::InverseGamma) = 1.0 / rand(Gamma(d.shape, d.scale))

function rand!(d::InverseGamma, A::Array{Float64})
    rand!(Gamma(d.shape, d.scale), A)
    for i in 1:length(A)
    	A[i] = 1.0 / A[i]
    end
    A
end
