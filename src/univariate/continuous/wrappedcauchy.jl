"""
    WrappedCauchy(r)

The Wrapped Cauchy distribution with scale factor `r` has probability density function

```math
f(x; r) = \\frac{1-r^2}{2\\pi(1+r^2-2r\\cos(x-\\mu))}, \\quad x \\in [-\\pi, \\pi].
```

```julia
WrappedCauchy(μ,r)   # Wrapped Cauchy distribution centered on μ with scale factor r

params(d)       # Get the location and scale parameters, i.e. (μ, r)
```

External links

* [Wrapped Cauchy distribution on Wikipedia](https://en.wikipedia.org/wiki/Wrapped_Cauchy_distribution)
"""
struct WrappedCauchy{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    r::T
    WrappedCauchy{T}(μ::T, r::T) where {T <: Real} = new{T}(μ, r)
end


function WrappedCauchy(μ::T, r::T; check_args::Bool=true) where {T <: Real}
    @check_args WrappedCauchy (μ, -π < μ < π) (r, zero(r) < r < one(r))

    return WrappedCauchy{T}(μ, r)
end

WrappedCauchy(μ::Real, r::Real; check_args::Bool=true) = WrappedCauchy(promote(μ, r)...; check_args=check_args)
WrappedCauchy(μ::Integer, r::Integer; check_args::Bool=true) = WrappedCauchy(float(μ), float(r); check_args=check_args)
WrappedCauchy(r::Real=0.0) = WrappedCauchy(zero(r), r; check_args=false)

@distr_support WrappedCauchy -oftype(d.r, π) oftype(d.r, π)


params(d::WrappedCauchy) = (d.μ, d.r)
partype(::WrappedCauchy{T}) where {T} = T

location(d::WrappedCauchy) = d.μ
scale(d::WrappedCauchy) = d.r

#### Statistics

mean(d::WrappedCauchy) = d.μ

var(d::WrappedCauchy) = one(d.r) - d.r

skewness(d::WrappedCauchy) = zero(d.r)

median(d::WrappedCauchy) = zero(d.r)

mode(d::WrappedCauchy) = zero(d.r)

entropy(d::WrappedCauchy) = log2π + log1p(-d.r^2)


cf(d::WrappedCauchy, t::Real) = cis(t * d.μ - abs(t) * log(d.r) * im)



function pdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    res = inv2π * ((1 - r^2) / (1 + r^2 - 2 * r * cos(x - μ)))
    return insupport(d, x) ? res : zero(res)
end

function logpdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    res = - log1p(2 * r * (r - cos(x-μ)) / (1 - r^2)) - log2π
    return insupport(d, x) ? res : oftype(res, -Inf)
end

function cdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    min_d, max_d = extrema(d)
    c = (one(r) + r) / (one(r) - r)
    res = (atan(c * tan(mod2pi(x - μ) / 2)) / π
    return if x < min_d
        zero(res)
    elseif x > max_d
        one(res)
    elseif res < 0 # if mod2pi(x - μ) > π
        1 + res
    else
        res
    end
end

function rand(rng::AbstractRNG, d::WrappedCauchy)
    return mod2pi(d.μ + log(d.r) * tan(π * (rand(rng) - 0.5))) - π

end
