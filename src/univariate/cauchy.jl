immutable Cauchy <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Cauchy(l::Real, s::Real)
	    if s > 0.0
	    	new(float64(l), float64(s))
	    else
	    	error("scale must be positive")
	    end
	end
end

Cauchy(l::Real) = Cauchy(l, 1.0)
Cauchy() = Cauchy(0.0, 1.0)

@_jl_dist_2p Cauchy cauchy

entropy(d::Cauchy) = log(d.scale) + log(4.0 * pi)

insupport(::Cauchy, x::Real) = isfinite(x)
insupport(::Type{Cauchy}, x::Real) = isfinite(x)

kurtosis(d::Cauchy) = NaN

mean(d::Cauchy) = NaN

median(d::Cauchy) = d.location

mgf(d::Cauchy, t::Real) = NaN

function cf(d::Cauchy, t::Real)
	m, theta = d.location, d.scale
	return exp(im * t * m - theta * abs(t))
end

modes(d::Cauchy) = [d.location]

skewness(d::Cauchy) = NaN

var(d::Cauchy) = NaN

function fit_mle{T <: Real}(::Type{Cauchy}, x::Array{T})
	c = median(x)
	l, u = iqr(x)
	return Cauchy(c, (u - l) / 2.0)
end
