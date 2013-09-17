immutable Cauchy <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Cauchy(l::Real, s::Real)
        s > zero(s) || error("scale must be positive")
	new(float64(l), float64(s))
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
    exp(im * t * d.location - d.scale * abs(t))
end

mode(d::Cauchy) = d.location
modes(d::Cauchy) = [mode(d)]

skewness(d::Cauchy) = NaN

var(d::Cauchy) = NaN

# Note: this is not a Maximum Likelihood estimator
function fit{T <: Real}(::Type{Cauchy}, x::Array{T})
    l, u = iqr(x)
    Cauchy(median(x), (u - l) / 2.0)
end

location(d::Cauchy) = d.location
scale(d::Cauchy) = d.scale