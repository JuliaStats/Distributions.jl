# Truncated pareto distribution
"""
    TruncatedPareto(α, θ, u)

The *truncated Pareto distribution* with shape `α`, scale `θ` and upper limit `u` has the following p.d.f.:

```math
f(x; \\alpha; \\theta; u) = \\frac{\\alpha u^\\alpha x^{-\\alpha - 1}}{1-\\left(\\frac{\\theta}{u}\\right)^\\alpha}
```
```julia
truncated(Pareto(),u)         # Pareto distribution with unit shape and unit scale, i.e. Pareto(1, 1) truncated at u.
truncated(Pareto(a), u)       # Pareto distribution with shape a and unit scale, i.e. Pareto(a, 1) truncated at u.
truncated(Pareto(a, b), u)    # Pareto distribution with shape a and scale b, truncated at u.

params(d)       # Get the parameters, i.e. (a, b, u)
shape(d)        # Get the shape parameter, i.e. a
scale(d)        # Get the scale parameter, i.e. b
upper(d)        # Get the upper limit parameter, i.e. u
```

External links
* [Truncated Pareto distribution on Wikipedia](https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution)

"""
function truncated(d::Pareto, u::T) where {T <: Real}
    return truncated(d, d.θ, u)
end

minimum(d::Truncated{Pareto{T},Continuous}) where {T <: Real} = d.lower
maximum(d::Truncated{Pareto{T},Continuous}) where {T <: Real} = d.upper

function _kmom(α::T, θ::T, u::T, k::Int) where {T <: Real}
    """
    Returns the k'th moment of a truncated Pareto distribution.

    Source: Clark, David R. (2013) - A Note on the Upper-Truncated Pareto Distribution
            in Enterprise Risk Management Symposium - April 22-24, 2013
    """
    p = θ/u
    return (α*θ^k/(α-k)) * (1-p^(α-k))/(1-p^α)
end

function mean(d::Truncated{Pareto{T},Continuous}) where {T <: Real}
    d0 = d.untruncated
    α = d0.α
    θ = d0.θ
    u = d.upper
    if α == 1:
        return ((u*θ)/(u-θ)) * log(u/θ)
    end
    return _kmom(α, θ, u, 1)
    #return (θ^α)/(1-(θ/u)^α) * α/(α-1) * (θ^(1-α) - u^(1-α))
end

function var(d::Truncated{Pareto{T},Continuous}) where {T <: Real}
    d0 = d.untruncated
    α = d0.α
    θ = d0.θ
    u = d.upper
    return _kmom(α, θ, u, 2) - _kmom(α, θ, u, 1)^2
end
