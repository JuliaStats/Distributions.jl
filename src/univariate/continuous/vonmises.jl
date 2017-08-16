"""
    VonMises(μ, κ)

The *von Mises distribution* with mean `μ` and concentration `κ` has probability density function

```math
f(x; \\mu, \\kappa) = \\frac{1}{2 \\pi I_0(\\kappa)} \\exp \\left( \\kappa \\cos (x - \\mu) \\right)
```

```julia
VonMises()       # von Mises distribution with zero mean and unit concentration
VonMises(κ)      # von Mises distribution with zero mean and concentration κ
VonMises(μ, κ)   # von Mises distribution with mean μ and concentration κ
```

External links

* [von Mises distribution on Wikipedia](http://en.wikipedia.org/wiki/Von_Mises_distribution)

"""
struct VonMises{T<:Real} <: ContinuousUnivariateDistribution
    μ::T      # mean
    κ::T      # concentration
    I0κ::T    # I0(κ), where I0 is the modified Bessel function of order 0

    function VonMises{T}(μ::T, κ::T) where T
        @check_args(VonMises, κ > zero(κ))
        new{T}(μ, κ, besseli(zero(T), κ))
    end
end

VonMises(μ::T, κ::T) where {T<:Real} = VonMises{T}(μ, κ)
VonMises(μ::Real, κ::Real) = VonMises(promote(μ, κ)...)
VonMises(μ::Integer, κ::Integer) = VonMises(Float64(μ), Float64(κ))
VonMises(κ::Real) = VonMises(0.0, κ)
VonMises() = VonMises(0.0, 1.0)

show(io::IO, d::VonMises) = show(io, d, (:μ, :κ))

@distr_support VonMises d.μ - π d.μ + π

#### Conversions

convert(::Type{VonMises{T}}, μ::Real, κ::Real) where {T<:Real} = VonMises(T(μ), T(κ))
convert(::Type{VonMises{T}}, d::VonMises{S}) where {T<:Real, S<:Real} = VonMises(T(d.μ), T(d.κ))

#### Parameters

params(d::VonMises) = (d.μ, d.κ)
@inline partype(d::VonMises{T}) where {T<:Real} = T


#### Statistics

mean(d::VonMises) = d.μ
median(d::VonMises) = d.μ
mode(d::VonMises) = d.μ
var(d::VonMises) = 1 - besseli(1, d.κ) / d.I0κ
# deprecated 12 September 2016
@deprecate circvar(d) var(d)
entropy(d::VonMises) = log(twoπ * d.I0κ) - d.κ * (besseli(1, d.κ) / d.I0κ)

cf(d::VonMises, t::Real) = (besseli(abs(t), d.κ) / d.I0κ) * cis(t * d.μ)


#### Evaluations

pdf(d::VonMises, x::Real) = exp(d.κ * cos(x - d.μ)) / (twoπ * d.I0κ)
logpdf(d::VonMises, x::Real) = d.κ * cos(x - d.μ) - log(d.I0κ) - log2π

cdf(d::VonMises, x::Real) = _vmcdf(d.κ, d.I0κ, x - d.μ, 1e-15)

function _vmcdf(κ::Real, I0κ::Real, x::Real, tol::Real)
    j = 1
    cj = besseli(j, κ) / j
    s = cj * sin(j * x)
    while abs(cj) > tol
        j += 1
        cj = besseli(j, κ) / j
        s += cj * sin(j * x)
    end
    return (x + 2s / I0κ) / twoπ + 1//2
end


#### Sampling

sampler(d::VonMises) = VonMisesSampler(d.μ, d.κ)
