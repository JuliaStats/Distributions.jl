# Truncated pareto distribution
"""
    TruncatedPareto(α, θ, ν)

The *truncated Pareto distribution* with shape `α`, scale `θ` and upper limit `u` has the following p.d.f.:

```math
f(x; \\alpha; \\theta; \\nu) = \\frac{\\alpha \\nu^\\alpha x^{-\\alpha - 1}}{1-\\left(\\frac{\\theta}{\\nu}\\right)^\\alpha}
```
```julia
truncated(Pareto(), ν)         # Pareto distribution with unit shape and unit scale, i.e. Pareto(1, 1) truncated at u.
truncated(Pareto(a), ν)       # Pareto distribution with shape a and unit scale, i.e. Pareto(a, 1) truncated at u.
truncated(Pareto(a, b), ν)    # Pareto distribution with shape a and scale b, truncated at u.

params(d)       # Get the parameters, i.e. (a, b, ν)
shape(d)        # Get the shape parameter, i.e. a
scale(d)        # Get the scale parameter, i.e. b
upper(d)        # Get the upper limit parameter, i.e. ν
```

External links
* [Truncated Pareto distribution on Wikipedia](https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution)

"""
function truncated(d::Pareto, ν::T) where {T <: Real}
    return truncated(d, d.θ, ν)
end

minimum(d::Truncated{Pareto{T},Continuous}) where {T <: Real} = d.lower
maximum(d::Truncated{Pareto{T},Continuous}) where {T <: Real} = d.upper

function cdf(d::Truncated{Pareto{T},Continuous},x::T) where {T <: Real}
    d0 = d.untruncated
    x <= d.lower ? zero(T) :
    x >= d.upper ? one(T) :
    (1-(d.lower^d0.α) * x^(-d0.α)) / (1-(d.lower/d.upper)^α)
end

function logcdf(d::Truncated{Pareto{T},Continuous},x::T) where {T <: Real}
    d0 = d.untruncated
    x <= d.lower ? T(-Inf) :
    x >= d.upper ? zero(T) :
    log(1-(d.lower^d0.α) + log(x)*(-d0.α)) - log(1-(d.lower/d.upper)^α)
end

function pdf(d::Truncated{Pareto{T},Continuous},x::T) where {T <: Real}
    d0 = d.untruncated
    x <= d.lower ? zero(T) :
    x >= d.upper ? zero(T) :
    α*d.lower^d0.α*x^(-d0.α-1) / (1-(d.lower/d.upper)^α)
end

function logpdf(d::Truncated{Pareto{T},Continuous},x::T) where {T <: Real}
    d0 = d.untruncated
    x <= d.lower ? zero(T) :
    x >= d.upper ? zero(T) :
    log(α)+d0.α*log(d.lower)+log(x)*(-d0.α-1) - log(1-(d.lower/d.upper)^α)
end

function quantile(d::Truncated{Pareto{T},Continuous}, p::Real) where {T <: Real}
    d0 = d.untruncated
    α = d0.α
    θ = d0.θ
    ν = d.upper

    θ(1-p*(1-(θ/ν)^α))^-(1/α)
end

# Sampling

function rand(rng::AbstractRNG, d::Truncated{Pareto{T},Continuous}) where {T <: Real}
    α = d.untruncated.α
    ν = d.upper
    θ = d.lower
    (rand(rng)*ν^α+θ^α*rand(rng)+ν/(ν*θ)^α)^(-1/α)
end

# Moments

function _kmom(α::T, θ::T, ν::T, k::Int) where {T <: Real}
    """
    Returns the k'th moment of a truncated Pareto distribution.
    Not exported, used in mean() and variance().
    Source: Clark, David R. (2013) - A Note on the Upper-Truncated Pareto Distribution
            in Enterprise Risk Management Symposium - April 22-24, 2013
    """
    p = θ/ν
    ((α == k) || (α == 0)) ? log(p^-1) : (α*θ^k/(α-k)) * (1-p^(α-k))/(1-p^α)
end

function mean(d::Truncated{Pareto{T},Continuous}) where {T <: Real}
    d0 = d.untruncated
    α = d0.α
    θ = d0.θ
    ν = d.upper
    α == 1 ? ((ν*θ)/(ν-θ)) * log(ν/θ) : _kmom(α, θ, ν, 1)
    #return (θ^α)/(1-(θ/u)^α) * α/(α-1) * (θ^(1-α) - u^(1-α))
end

function var(d::Truncated{Pareto{T},Continuous}) where {T <: Real}
    d0 = d.untruncated
    α = d0.α
    θ = d0.θ
    ν = d.upper
    return _kmom(α, θ, ν, 1) * _kmom(α-1, θ, ν, 1)  - _kmom(α, θ, ν, 1)^2
end

function median(d::Truncated{Pareto{T},Continuous}) where {T <: Real}
    quantile(d, 0.5)
end
