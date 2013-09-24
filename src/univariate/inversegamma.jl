# Inverse Gamma distribution

immutable InverseGamma <: ContinuousUnivariateDistribution
    shape::Float64 # location
    scale::Float64 # scale
    function InverseGamma(sh::Real, sc::Real)
    	sh > zero(sh) && sc > zero(sc) || error("Both shape and scale must be positive")
    	new(float64(sh), float64(sc))
    end
end

@continuous_distr_support InverseGamma 0.0 Inf

_inv(d::InverseGamma) = Gamma(d.shape, 1.0 / d.scale)

scale(d::InverseGamma) = d.scale
rate(d::InverseGamma) = 1.0 / d.scale

mean(d::InverseGamma) = d.shape > 1.0 ? d.scale / (d.shape - 1.0) : Inf
var(d::InverseGamma) = d.shape > 2.0 ? abs2(d.scale) / (abs2(d.shape - 1.0) * (d.shape - 2.0)) : Inf
skewness(d::InverseGamma) = d.shape > 3.0 ? (4.0 * sqrt(d.shape - 2.0)) / (d.shape - 3.0) : NaN
kurtosis(d::InverseGamma) = d.shape > 4.0 ? (30.0 * d.shape - 66.0) / ((d.shape - 3.0) * (d.shape - 4.0)) : NaN

mode(d::InverseGamma) = d.scale / (d.shape + 1.0)
modes(d::InverseGamma) = [mode(d)]

cdf(d::InverseGamma, x::Real) = ccdf(_inv(d), 1.0 / x)
ccdf(d::InverseGamma, x::Real) = cdf(_inv(d), 1.0 / x)
logcdf(d::InverseGamma, x::Real) = logccdf(_inv(d), 1.0 / x)
logccdf(d::InverseGamma, x::Real) = logcdf(_inv(d), 1.0 / x)

quantile(d::InverseGamma, p::Real) = 1.0 / cquantile(_inv(d), p)
cquantile(d::InverseGamma, p::Real) = 1.0 / quantile(_inv(d), p)
invlogcdf(d::InverseGamma, p::Real) = 1.0 / invlogccdf(_inv(d), p)
invlogccdf(d::InverseGamma, p::Real) = 1.0 / invlogcdf(_inv(d), p)

function entropy(d::InverseGamma)
    a = d.shape
    b = d.scale
    a + log(b) + lgamma(a) - (1.0 + a) * digamma(a)
end


function mgf(d::InverseGamma, t::Real)
    a = d.shape
    b = d.scale
    (2 * (-b * t)^(a / 2)) / gamma(a) * besselk(a, sqrt(-4.0 * b * t))
end

function cf(d::InverseGamma, t::Real)
    a = d.shape
    b = d.scale
    (2 * (-im * b * t)^(a / 2)) / gamma(a) * besselk(a, sqrt(-4.0 * im * b * t))
end

pdf(d::InverseGamma, x::Real) = exp(logpdf(d, x))

function logpdf(d::InverseGamma, x::Real)
    a = d.shape
    b = d.scale
    a * log(b) - lgamma(a) - (a + 1.0) * log(x) - b / x
end

rand(d::InverseGamma) = 1.0 / rand(_inv(d))

function rand!(d::InverseGamma, A::Array{Float64})
    rand!(_inv(d), A)
    for i in 1:length(A)
    	A[i] = 1.0 / A[i]
    end
    A
end
