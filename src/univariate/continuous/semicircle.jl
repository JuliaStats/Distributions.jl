"""
    Semicircle(r)

The Wigner semicircle distribution with radius `r`.
"""
struct Semicircle{T<:Real} <: ContinuousUnivariateDistribution
    r::T

    Semicircle{T}(r) where {T} = (@check_args(Semicircle, r > 0); new{T}(r))
end

Semicircle(r::Real) = Semicircle{typeof(r)}(r)
Semicircle(r::Integer) = Semicircle(Float64(r))

@distr_support Semicircle -d.r +d.r

params(d::Semicircle) = (d.r,)

mean(d::Semicircle) = zero(d.r)
var(d::Semicircle) = d.r^2 / 4
skewness(d::Semicircle) = zero(d.r)
median(d::Semicircle) = zero(d.r)
mode(d::Semicircle) = zero(d.r)
entropy(d::Semicircle) = log(π * d.r) - oftype(d.r, 0.5)

function pdf(d::Semicircle, x::Real)
    if abs(x) < d.r
        return 2 / (π * d.r^2) * sqrt(d.r^2 - x^2)
    else
        return 0.0
    end
end

function logpdf(d::Semicircle, x::Real)
    if abs(x) < d.r
        return log(2 / π) - 2 * log(d.r) + 1/2 * log(r^2 - x^2)
    else
        return oftype(x, Inf)
    end
end

function cdf(d::Semicircle, x::Real)
    if abs(x) < d.r
        u = x / d.r
        return (u * sqrt(1 - u^2) + asin(u)) / π + one(x) / 2
    elseif x ≥ d.r
        return one(x)
    else
        return zero(x)
    end
end
