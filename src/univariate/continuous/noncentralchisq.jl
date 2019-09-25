"""
    NoncentralChisq(ν, λ)

The *noncentral chi-squared distribution* with `ν` degrees of freedom and noncentrality parameter `λ` has the probability density function

```math
f(x; \\nu, \\lambda) = \\frac{1}{2} e^{-(x + \\lambda)/2} \\left( \\frac{x}{\\lambda} \\right)^{\\nu/4-1/2} I_{\\nu/2-1}(\\sqrt{\\lambda x}), \\quad x > 0
```

It is the distribution of the sum of squares of `ν` independent [`Normal`](@ref) variates with individual means ``\\mu_i`` and

```math
\\lambda = \\sum_{i=1}^\\nu \\mu_i^2
```

```julia
NoncentralChisq(ν, λ)     # Noncentral chi-squared distribution with ν degrees of freedom and noncentrality parameter λ

params(d)    # Get the parameters, i.e. (ν, λ)
```

External links

* [Noncentral chi-squared distribution on Wikipedia](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution)
"""
struct NoncentralChisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    λ::T
    NoncentralChisq{T}(ν::T, λ::T) where {T <: Real} = new{T}(ν, λ)
end

function NoncentralChisq(ν::T, λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(NoncentralChisq, ν > zero(ν) && λ >= zero(λ))
    return NoncentralChisq{T}(ν, λ)
end

NoncentralChisq(ν::Real, λ::Real) = NoncentralChisq(promote(ν, λ)...)
NoncentralChisq(ν::Integer, λ::Integer) = NoncentralChisq(float(ν), float(λ))

@distr_support NoncentralChisq 0.0 Inf

#### Conversions

function convert(::Type{NoncentralChisq{T}}, ν::S, λ::S) where {T <: Real, S <: Real}
    NoncentralChisq(T(ν), T(λ))
end
function convert(::Type{NoncentralChisq{T}}, d::NoncentralChisq{S}) where {T <: Real, S <: Real}
    NoncentralChisq(T(d.ν), T(d.λ), check_args=false)
end

### Parameters

params(d::NoncentralChisq) = (d.ν, d.λ)
@inline partype(d::NoncentralChisq{T}) where {T<:Real} = T


### Statistics

mean(d::NoncentralChisq) = d.ν + d.λ
var(d::NoncentralChisq) = 2(d.ν + 2d.λ)
skewness(d::NoncentralChisq) = 2sqrt2*(d.ν + 3d.λ)/sqrt(d.ν + 2d.λ)^3
kurtosis(d::NoncentralChisq) = 12(d.ν + 4d.λ)/(d.ν + 2d.λ)^2

function mgf(d::NoncentralChisq, t::Real)
    exp(d.λ * t/(1 - 2t))*(1 - 2t)^(-d.ν/2)
end

function cf(d::NoncentralChisq, t::Real)
    cis(d.λ * t/(1 - 2im*t))*(1 - 2im*t)^(-d.ν/2)
end


### Evaluation & Sampling

@_delegate_statsfuns NoncentralChisq nchisq ν λ

# TODO: remove RFunctions dependency
@rand_rdist(NoncentralChisq)
rand(d::NoncentralChisq) = StatsFuns.RFunctions.nchisqrand(d.ν, d.λ)
