"""
    MaxwellBoltzmann(a)
The *Maxwell-Boltzmann distribution* is the chi distribution with three degrees of freedom and scale parameter `a`. The scale parameter is related to the mass and temperature of particles a = √(kT/m) where k is Boltzmann's constant, T is the temperature of the gas in Kelvin, and m is the mass of the particle in kilograms.

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

MaxwellBoltmann() = MaxwellBoltzmann(1.0)

MaxwellBoltzmann(a::Integer) = MaxwellBoltzmann(float(a))

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

@_delegate_statsfun MaxwellBoltzmann maxwellboltzmann a

pdf(d::MaxwellBoltzmann, x::Real) = sqrt(2 / π) * x^2 * exp(-x^2 / (2d.a^2)) / d.a^3
cdf(d::MaxwellBoltzmann, x::Real) = erf(x / (sqrt(2) * d.a)) - sqrt(2 / π) * x * exp(-x^2 / 2d.a^2) / d.a
