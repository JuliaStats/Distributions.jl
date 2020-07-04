"""
    MaxwellBoltzmann(a)

The *Maxwell-Boltzmann distribution* has probability density function
```math
f(x; a) = \\sqrt{\\frac{2}{\\pi}}\\frac{x^2}{a^3}\\exp\\left(-\\frac{x^2}{2a^2}\\right)
```
where
```math
a = \\sqrt{\\frac{kT}{m}}
```
`k` is the Boltzmann constant and `m` is the particle mass.

```julia
MaxwellBoltzmann()        # Maxwell-Boltzmann distribution with a = 1
MaxwellBoltzmann(a)       # Maxwell-Boltzmann distribution with chosen parameter a
MaxwellBoltzmann(T, m)    # Maxwell-Boltzmann distribution with chosen temperature T and mass m
```

External links

* [Maxwell-Boltzmann distribution on Wikipedia](https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution)
"""
struct MaxwellBoltzmann{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    MaxwellBoltzmann{T}(a::T) where {T} = new{T}(a)
end

function MaxwellBoltzmann(a::T; check_args=true) where {T<:Real}
    check_args && @check_args(MaxwellBoltzmann, a > zero(a))
    return MaxwellBoltzmann{T}(a)
end

MaxwellBoltzmann() = MaxwellBoltzmann(1.0)
MaxwellBoltzmann(temp::T, mass::T) where {T<:Real} = MaxwellBoltzmann(sqrt(ustrip(k_B) * temp / mass))

@distr_support MaxwellBoltzmann 0.0 Inf

#### Parameters

dof(d::MaxwellBoltzmann) = 3
params(d::MaxwellBoltzmann) = (d.a,)
@inline partype(d::MaxwellBoltzmann{T}) where {T<:Real} = T

#### Conversions

convert(::Type{MaxwellBoltzmann{T}}, a::Real) where {T<:Real} = MaxwellBoltzmann(T(a))
convert(::Type{MaxwellBoltzmann{T}}, d::MaxwellBoltzmann{S}) where {T<:Real, S<:Real} = MaxwellBoltzmann(T(d.a))

#### Statistics

mean(d::MaxwellBoltzmann) = 2d.a * sqrt(2 / π)
var(d::MaxwellBoltzmann) = d.a^2(3π - 8) / π
skewness(d::MaxwellBoltzmann) = 2 * sqrt(2) * (16 - 5π) / (3π - 8)^(3/2)
kurtosis(d::MaxwellBoltzmann) = 4 * (-96 + 40π - 3π^2) / (3π - 8)^2
mode(d::MaxwellBoltzmann) = sqrt(2) * d.a
entropy(d::MaxwellBoltzmann) = log(d.a * sqrt(2π)) + (-digamma(1)) - 0.5

#### Evaluation

pdf(d::MaxwellBoltzmann, x::Real) = sqrt(2 / π) * x^2 * exp(-x^2 / (2d.a^2)) / d.a^3
cdf(d::MaxwellBoltzmann, x::Real) = erf(x / (sqrt(2) * d.a)) - sqrt(2 / π) * x * exp(-x^2 / 2d.a^2) / d.a
