##############################################################################
#
# InvertedGamma distribution, cf Bayesian Theory (Barnardo & Smith) p 119
#
#  Note that B&S parametrize in terms of shape/rate, but this is uses
#  shape/scale so that it is consistent with the implementation of Gamma
#
##############################################################################

immutable InvertedGamma <: ContinuousUnivariateDistribution
    shape::Float64 # location
    scale::Float64 # scale
    function InvertedGamma(sh::Real, sc::Real)
    	sh > zero(sh) && sc > zero(sc) || error("Both shape and scale must be positive")
    	new(float64(sh), float64(sc))
    end
end

insupport(::InvertedGamma, x::Real) = zero(x) <= x < Inf
insupport(::Type{InvertedGamma}, x::Real) = zero(x) <= x < Inf


mean(d::InvertedGamma) = d.shape > 1.0 ? (1.0 / d.scale) / (d.shape - 1.0) : Inf
var(d::InvertedGamma) = d.shape > 2.0 ? (1.0 / d.scale)^2 / ((d.shape - 1.0)^2 * (d.shape - 2.0)) : Inf
skewness(d::InvertedGamma) = d.shape > 3.0 ? (4.0 * sqrt(d.shape - 2.0)) / (d.shape - 3.0) : NaN
kurtosis(d::InvertedGamma) = d.shape > 4.0 ? (30.0 * d.shape - 66.0) / ((d.shape - 3.0) * (d.shape - 4.0)) : NaN

mode(d::InvertedGamma) = (1.0 / d.scale) / (d.shape + 1.0)
modes(d::InvertedGamma) = [mode(d)]

cdf(d::InvertedGamma, x::Real) = ccdf(Gamma(d.shape, d.scale), 1.0 / x)
ccdf(d::InvertedGamma, x::Real) = cdf(Gamma(d.shape, d.scale), 1.0 / x)
logcdf(d::InvertedGamma, x::Real) = logccdf(Gamma(d.shape, d.scale), 1.0 / x)
logccdf(d::InvertedGamma, x::Real) = logcdf(Gamma(d.shape, d.scale), 1.0 / x)

quantile(d::InvertedGamma, p::Real) = 1.0 / cquantile(Gamma(d.shape, d.scale), p)
cquantile(d::InvertedGamma, p::Real) = 1.0 / quantile(Gamma(d.shape, d.scale), p)
invlogcdf(d::InvertedGamma, p::Real) = 1.0 / invlogccdf(Gamma(d.shape, d.scale), p)
invlogccdf(d::InvertedGamma, p::Real) = 1.0 / invlogcdf(Gamma(d.shape, d.scale), p)

function entropy(d::InvertedGamma)
    a, b = d.shape, d.scale
    a - log(b) + lgamma(a) - (1.0 + a) * digamma(a)
end


function mgf(d::InvertedGamma, t::Real)
    a, b = d.shape, d.scale
    (2 * (-b * t)^(a / 2)) / gamma(a) * besselk(a, sqrt(-4.0 * b * t))
end

function cf(d::InvertedGamma, t::Real)
    a, b = d.shape, d.scale
    (2 * (-im * b * t)^(a / 2)) / gamma(a) * besselk(a, sqrt(-4.0 * im * b * t))
end

pdf(d::InvertedGamma, x::Real) = exp(logpdf(d, x))

function logpdf(d::InvertedGamma, x::Real)
    a, b = d.shape, d.scale
    -(a * log(b)) - lgamma(a) - ((a + 1.0) * log(x)) - 1.0/ (b * x)
end

rand(d::InvertedGamma) = 1.0 / rand(Gamma(d.shape, d.scale))

function rand!(d::InvertedGamma, A::Array{Float64})
    rand!(Gamma(d.shape, d.scale), A)
    for i in 1:length(A)
    	A[i] = 1.0 / A[i]
    end
    A
end
