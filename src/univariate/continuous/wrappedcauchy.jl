"""
    WrappedCauchy(r)
The Wrapped Cauchy distribution with scale factor `r` has probability density function
```math
f(x; r) = \\frac{1-r^2}{2\\pi(1+r^2-2r\\cos(x))}, \\quad x \\in [0, 2\pi].
```
```julia
WrappedCauchy(r)   # Wrapped Cauchy distribution with scale factor r
params(d)       # Get the radius parameter, i.e. (r,)
```
External links
* [Wigner semicircle distribution on Wikipedia](https://en.wikipedia.org/wiki/Wigner_semicircle_distribution)
"""
struct WrappedCauchy{T<:Real} <: ContinuousUnivariateDistribution
    r::T
    WrappedCauchy{T}(r::T) where {T <: Real} = new{T}(r)
end


function WrappedCauchy(r::Real; check_args::Bool=true)
    @check_args WrappedCauchy (r, r > zero(r), r < one(r))
    return WrappedCauchy{typeof(r)}(r)
end

WrappedCauchy(r::Integer; check_args::Bool=true) = WrappedCauchy(float(r); check_args=check_args)

@distr_support WrappedCauchy -π +π

params(d::WrappedCauchy) = (d.r,)

mean(d::WrappedCauchy) = zero(d.r)
var(d::WrappedCauchy) = one(d.r) - d.r
skewness(d::WrappedCauchy) = zero(d.r)
median(d::WrappedCauchy) = zero(d.r)
mode(d::WrappedCauchy) = zero(d.r)
entropy(d::WrappedCauchy) = log(2π * (one(d.r)-d.r^2))

function pdf(d::WrappedCauchy, x::Real)
    xx, r = promote(x, float(d.r))
    if insupport(d, xx)
        return (1-r^2) / (1 + r^2 - 2 * r * cos(xx)) / 2π
    else
        return oftype(r, 0)
    end
end

function logpdf(d::WrappedCauchy, x::Real)
    xx, r = promote(x, float(d.r))
    if insupport(d, xx)
        return log(π * (one(r) - r^2)) - log(oftype(r, 2) + 2 * r^2 - 4 * r * cos(xx))
    else
        return oftype(r, -Inf)
    end
end

function cdf(d::WrappedCauchy, x::Real)
    xx, r = promote(x, float(d.r))
    if insupport(d, xx)
        c = (one(r) + r) / (one(r) - r)
        return oftype(r,0.5) + atan(c * tan(xx / 2)) / π
    elseif x < minimum(d)
        return zero(r)
    else
        return one(r)
    end
end

function rand(rng::AbstractRNG, d::WrappedCauchy)
    return mod(π - log(d.r) * tan(π * (rand(rng) - oftype(r, 0.5))), 2π) - π
end

@quantile_newton WrappedCauchy