"""
    Semicircle(r)

The Wigner semicircle distribution with radius parameter `r` has probability
density function

```math
f(x; r) = \\frac{2}{\\pi r^2} \\sqrt{r^2 - x^2}, \\quad x \\in [-r, r].
```

```julia
Semicircle(r)   # Wigner semicircle distribution with radius r

params(d)       # Get the radius parameter, i.e. (r,)
```

External links

* [Wigner semicircle distribution on Wikipedia](https://en.wikipedia.org/wiki/Wigner_semicircle_distribution)
"""
struct Semicircle{T<:Real} <: ContinuousUnivariateDistribution
    r::T
    Semicircle{T}(r::T) where {T <: Real} = new{T}(r)
end


function Semicircle(r::T; check_args=true) where {T <: Real}
    check_args && @check_args(Semicircle, r > 0)
    return Semicircle{T}(r)
end

Semicircle(r::Integer) = Semicircle(float(r))

@distr_support Semicircle -d.r +d.r

params(d::Semicircle) = (d.r,)

mean(d::Semicircle) = zero(d.r)
var(d::Semicircle) = d.r^2 / 4
skewness(d::Semicircle) = zero(d.r)
median(d::Semicircle) = zero(d.r)
mode(d::Semicircle) = zero(d.r)
entropy(d::Semicircle) = log(π * d.r) - oftype(d.r, 0.5)

function pdf(d::Semicircle, x::Real)
    xx, r = promote(x, float(d.r))
    if insupport(d, xx)
        return 2 / (π * r^2) * sqrt(r^2 - xx^2)
    else
        return oftype(r, 0)
    end
end

function logpdf(d::Semicircle, x::Real)
    xx, r = promote(x, float(d.r))
    if insupport(d, xx)
        return log(oftype(r, 2) / π) - 2 * log(r) + log(r^2 - xx^2) / 2
    else
        return oftype(r, -Inf)
    end
end

function cdf(d::Semicircle, x::Real)
    xx, r = promote(x, float(d.r))
    if insupport(d, xx)
        u = xx / r
        return (u * sqrt(1 - u^2) + asin(u)) / π + one(xx) / 2
    elseif x < minimum(d)
        return zero(r)
    else
        return one(r)
    end
end

@quantile_newton Semicircle
