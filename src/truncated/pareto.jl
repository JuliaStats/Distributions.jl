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

minimum(d::Truncated{Pareto{T},Continuous}) where {T <: Real} = d.lower
maximum(d::Truncated{Pareto{T},Continuous}) where {T <: Real} = d.upper

truncated(d::Pareto, u::T) where T{ <: Real} = truncated(d::Pareto, d.θ, u)
