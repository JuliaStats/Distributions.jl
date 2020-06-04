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
    I0κx::T   # I0(κ) * exp(-κ), where I0 is the modified Bessel function of order 0
end

function VonMises(μ::T, κ::T; check_args=true) where {T <: Real}
    check_args && @check_args(VonMises, κ > zero(κ))
    return VonMises{T}(μ, κ, besselix(zero(T), κ))
end

VonMises(μ::Real, κ::Real) = VonMises(promote(μ, κ)...)
VonMises(μ::Integer, κ::Integer) = VonMises(float(μ), float(κ))
VonMises(κ::T) where {T <: Real} = VonMises(zero(T), κ)
VonMises() = VonMises(0.0, 1.0, check_args=false)

show(io::IO, d::VonMises) = show(io, d, (:μ, :κ))

@distr_support VonMises d.μ - π d.μ + π

#### Conversions

convert(::Type{VonMises{T}}, μ::Real, κ::Real) where {T<:Real} = VonMises(T(μ), T(κ))
convert(::Type{VonMises{T}}, d::VonMises{S}) where {T<:Real, S<:Real} = VonMises(T(d.μ), T(d.κ), check_args=false)

#### Parameters

params(d::VonMises) = (d.μ, d.κ)
partype(::VonMises{T}) where {T<:Real} = T


#### Statistics

mean(d::VonMises) = d.μ
median(d::VonMises) = d.μ
mode(d::VonMises) = d.μ
var(d::VonMises) = 1 - besselix(1, d.κ) / d.I0κx
# deprecated 12 September 2016
@deprecate circvar(d) var(d)
entropy(d::VonMises) = log(twoπ * d.I0κx) + d.κ * (1 - besselix(1, d.κ) / d.I0κx)

cf(d::VonMises, t::Real) = (besselix(abs(t), d.κ) / d.I0κx) * cis(t * d.μ)


#### Evaluations

#pdf(d::VonMises, x::Real) = exp(d.κ * (cos(x - d.μ) - 1)) / (twoπ * d.I0κx)
pdf(d::VonMises{T}, x::Real) where T<:Real =
    minimum(d) ≤ x ≤ maximum(d) ? exp(d.κ * (cos(x - d.μ) - 1)) / (twoπ * d.I0κx) : zero(T)
logpdf(d::VonMises{T}, x::Real) where T<:Real =
    minimum(d) ≤ x ≤ maximum(d) ? d.κ * (cos(x - d.μ) - 1) - log(d.I0κx) - log2π : -T(Inf)

cdf(d::VonMises, x::Real) = _vmcdf(d.κ, d.I0κx, x - d.μ, 1e-15)

function _vmcdf(κ::Real, I0κx::Real, x::Real, tol::Real)
    tol *= exp(-κ)
    j = 1
    cj = besselix(j, κ) / j
    s = cj * sin(j * x)
    while abs(cj) > tol
        j += 1
        cj = besselix(j, κ) / j
        s += cj * sin(j * x)
    end
    return (x + 2s / I0κx) / twoπ + 1//2
end


#### Sampling

rand(rng::AbstractRNG, d::VonMises) = rand(rng, sampler(d))
sampler(d::VonMises) = VonMisesSampler(d.μ, d.κ)
