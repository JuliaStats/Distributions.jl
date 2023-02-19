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


function Semicircle(r::Real; check_args::Bool=true)
    @check_args Semicircle (r, r > zero(r))
    return Semicircle{typeof(r)}(r)
end

Semicircle(r::Integer; check_args::Bool=true) = Semicircle(float(r); check_args=check_args)

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

function rand(rng::AbstractRNG, d::Semicircle)
    # Idea:
    # sample polar coordinates r,θ
    # of point uniformly distributed on radius d.r half disk
    # project onto x axis
    θ = rand(rng) # multiple of π
    r = d.r * sqrt(rand(rng))
    return cospi(θ) * r
end

@quantile_newton Semicircle
