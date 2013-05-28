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
    	if sh > 0.0 && sc > 0.0
    		new(float64(sh), float64(sc))
    	else
    		error("Both shape and scale must be positive")
    	end
    end
end

cdf(d::InvertedGamma, x::Real) = 1.0 - cdf(Gamma(d.shape, d.scale), 1.0 / x)

function entropy(d::InvertedGamma)
    a = d.shape
    b = d.scale
    return a + log(b) + lgamma(a) - (1.0 + a) * digamma(a)
end

insupport(d::InvertedGamma, x::Number) = isreal(x) && isfinite(x) && 0 <= x

function kurtosis(d::InvertedGamma)
    a = d.shape
    if a > 4.0
        return (30.0 * a - 66.0) / ((a - 3.0) * (a - 4.0))
    else
        error("kurtosis not defined unless shape > 4")
    end
end

function mean(d::InvertedGamma)
	if d.shape > 1.0
		return (1.0 / d.scale) / (d.shape - 1.0)
	else
		error("Expectation only defined if shape > 1")
	end
end

function mgf(d::InvertedGamma, t::Real)
    a, b = d.shape, d.scale
    return (2 * (-b * t)^(a / 2)) / gamma(a) *
           besselk(a, sqrt(-4.0 * b * t))
end

function cf(d::InvertedGamma, t::Real)
    a, b = d.shape, d.scale
    return (2 * (-im * b * t)^(a / 2)) / gamma(a) *
           besselk(a, sqrt(-4.0 * im * b * t))
end

modes(d::InvertedGamma) = [(1.0 / d.scale) / (d.shape + 1.0)]

pdf(d::InvertedGamma, x::Real) = exp(logpdf(d, x))

function logpdf(d::InvertedGamma, x::Real)
    return -(d.shape * log(d.scale)) - lgamma(d.shape) -
           ((d.shape + 1.0) * log(x)) - 1.0/ (d.scale * x)
end

function quantile(d::InvertedGamma, p::Real)
    return 1.0 / quantile(Gamma(d.shape, d.scale), 1.0 - p)
end

rand(d::InvertedGamma) = 1.0 / rand(Gamma(d.shape, d.scale))

function rand!(d::InvertedGamma, A::Array{Float64})
    A = rand!(Gamma(d.shape, d.scale), A)
    for i in 1:length(A)
    	A[i] = 1.0 / A[i]
    end
    return A
end

function skewness(d::InvertedGamma)
    a = d.shape
    if a > 3.0
        return (4.0 * sqrt(a - 2.0)) / (a - 3.0)
    else
        error("skewness not defined unless shape > 3")
    end
end

function var(d::InvertedGamma)
	if d.shape > 2.0
		(1.0 / d.scale)^2 / ((d.shape - 1.0)^2 * (d.shape - 2.0))
	else
		error("Variance only defined if shape > 2")
	end
end
