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
    @check_args WrappedCauchy (μ, μ > oftype(μ,-π), μ < oftype(μ, π)) (r, r > zero(r), r < one(r))
    return WrappedCauchy{T}(μ, r)
end

WrappedCauchy(μ::Real, r::Real; check_args::Bool=true) = WrappedCauchy(promote(μ, r)...; check_args=check_args)
WrappedCauchy(μ::Integer, r::Integer; check_args::Bool=true) = WrappedCauchy(float(μ), float(r); check_args=check_args)
WrappedCauchy(r::Real=0.0) = WrappedCauchy(zero(r), r; check_args=false)

const WrappedLorentz = WrappedCauchy

@distr_support WrappedCauchy oftype(d.r,-π) oftype(d.r,+π)

params(d::WrappedCauchy) = (d.μ, d.r)
partype(::WrappedCauchy{T}) where {T} = T

location(d::WrappedCauchy) = d.μ
scale(d::WrappedCauchy) = d.r

#### Statistics

mean(d::WrappedCauchy) = d.μ

var(d::WrappedCauchy) = one(d.r) - d.r

# deprecated 12 September 2016
@deprecate circvar(d) var(d)

skewness(d::WrappedCauchy) = zero(d.r)

median(d::WrappedCauchy) = zero(d.r)

mode(d::WrappedCauchy) = zero(d.r)

entropy(d::WrappedCauchy) = log(2π * (one(d.r) - d.r^2))

cf(d::WrappedCauchy, t::Real) = exp(im * t * d.μ) * d.r ^ abs(t)


function pdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    if insupport(d, x)
        return (1-r^2) / (1 + r^2 - 2 * r * cos(x-μ)) / 2π
    else
        return zero(x)
    end
end

function logpdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    if insupport(d, x)
        return log(1 - r^2) - log(1 + r^2 - 2 * r * cos(x-μ)) - log(2π)
    else
        return oftype(r, -Inf)
    end
end

function cdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    if insupport(d, x)
        c = (one(r) + r) / (one(r) - r)
        res = (atan(c * tan((x - μ) / 2)) - atan(c * tan(-(μ + π) / 2))) / π
        if μ == zero(μ) || x < mod(μ, 2π) - π
            return res
        else
            return 1 + res
        end
    elseif x < minimum(d)
        return zero(r)
    else
        return one(r)
    end
end

function rand(rng::AbstractRNG, d::WrappedCauchy)
    return mod(π - log(d.r) * tan(π * (rand(rng) - oftype(r, 0.5))) + d.μ - π, 2π) - π
end
