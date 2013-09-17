immutable Weibull <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    function Weibull(sh::Real, sc::Real)
    	zero(sh) < sh && zero(sc) < sc || error("Both shape and scale must be positive")
    	new(float64(sh), float64(sc))
    end
end

Weibull(sh::Real) = Weibull(sh, 1.0)

@_jl_dist_2p Weibull weibull

function cdf(d::Weibull, x::Real)
	if 0.0 < x
		return 1.0 - exp(-((x / d.scale)^d.shape))
	else
		0.0
	end
end

function entropy(d::Weibull)
    k, l = d.shape, d.scale
    return ((k - 1.0) / k) * -digamma(1.0) + log(l / k) + 1.0
end

insupport(::Weibull, x::Real) = zero(x) <= x < Inf
insupport(::Type{Weibull}, x::Real) = zero(x) <= x < Inf

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

mean(d::Weibull) = d.scale * gamma(1.0 + 1.0 / d.shape)

median(d::Weibull) = d.scale * log(2.0)^(1.0 / d.shape)

function modes(d::Weibull)
    if d.shape <= 1.0
        return [0.0]
    else
        return [d.scale * ((d.shape - 1.0) / d.shape)^(1.0 / d.shape)]
    end
end

function skewness(d::Weibull)
    tmp = gamma(1.0 + 3.0 / d.shape) * d.scale^3
    tmp -= 3.0 * mean(d) * var(d)
    tmp -= mean(d)^3
    return tmp / std(d)^3
end

var(d::Weibull) = d.scale^2 * gamma(1.0 + 2.0 / d.shape) - mean(d)^2

function mode(d::Weibull)
    inv_k = 1.0 / d.shape
    d.shape > 1.0 ? d.scale * (1.0 - inv_k) ^ inv_k : 0.0
end

modes(d::Weibull) = [mode(d)]
