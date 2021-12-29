## exgaussian.jl
##
## This code was written for the Distributions.jl package of Julia, by Jeff & Caitlin Miller,
##  milleratotago@gmail.com & chadenmiller@gmail.com, December 29, 2021.
## No warranty is expressed or implied.
##

@doc raw"""
    Exgaussian(μ,σ,τ)

The *Exgaussian distribution* is the sum of a normal with mean `μ` and standard deviation `σ>0`, plus an independent exponential with mean `τ>0`. 
It has probability density function

```math
f(x; \mu, \sigma, \tau) = \frac{1}{\tau\sigma\sqrt{2\pi}}
   \exp\left( \mu/\tau + \sigma^2/(2\tau^2) \right)
   \Phi\left( (x-\mu-\sigma^2/\tau)/\sigma \right)
```
where $\Phi$ is the CDF of the standard Normal(0,1).

Note that unlike normal.jl, we require σ > 0, in addition to τ > 0.

```julia
Exgaussian(μ, σ, τ)   # Exgaussian distribution with normal mean μ, normal variance σ^2, and exponential mean τ

params(d)         # Get the parameters, i.e. (μ, σ, τ)
mean(d)           # Get the mean, i.e. μ+τ
std(d)            # Get the standard deviation, i.e. sqrt(σ^2+τ^2)
```

External links

* [Exgaussian distribution on Wikipedia](http://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)

"""
struct Exgaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    τ::T
    rate::T
    Exgaussian{T}(µ::T, σ::T, τ::T) where {T<:Real} = new{T}(µ, σ, τ, 1/τ)
end

function Exgaussian(μ::T, σ::T, τ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Exgaussian, σ > zero(σ) && τ > zero(τ))
    return Exgaussian{T}(μ, σ, τ)
end

#### Outer constructors
Exgaussian(μ::Real, σ::Real, τ::Real) = Exgaussian(promote(μ, σ, τ)...)
Exgaussian(μ::Integer, σ::Integer, τ::Real) = Exgaussian(float(μ), float(σ), float(τ))

# #### Conversions
convert(::Type{Exgaussian{T}}, μ::S, σ::S, τ::S) where {T <: Real, S <: Real} = Exgaussian(T(μ), T(σ), T(τ))
convert(::Type{Exgaussian{T}}, d::Exgaussian{S}) where {T <: Real, S <: Real} = Exgaussian(T(d.μ), T(d.σ), T(d.τ), check_args=false)

@distr_support Exgaussian -Inf Inf

#### Parameters

params(d::Exgaussian) = (d.μ, d.σ, d.τ)
@inline partype(d::Exgaussian{T}) where {T<:Real} = T

Base.eltype(::Type{Exgaussian{T}}) where {T} = T

#### Statistics

mean(d::Exgaussian) = d.μ+d.τ
var(d::Exgaussian) = abs2(d.σ) + abs2(d.τ)
skewness(d::Exgaussian{T}) where {T<:Real} = 2/(d.σ*d.rate)^3 * (1 + 1/(d.rate*d.σ)^2)^(-1.5)
kurtosis(d::Exgaussian{T}) where {T<:Real} = 3*(1+2/(d.σ*d.rate)^2 + 3/(d.σ*d.rate)^4) / (1 + 1/(d.σ*d.rate)^2)^2 - 3
kurtosis(d::Exgaussian, excess::Bool) = kurtosis(d) + (excess ? 0.0 : 3.0)

#### Evaluation

# pdf

function pdf(d::Exgaussian, x::Real)
    μ, σ, rate = d.μ, d.σ, d.rate
    t1 = -x*rate + μ*rate + 0.5*(σ*rate)^2
    t2 = (x - μ - σ^2*rate) / σ
    return rate*exp( t1 + _normlogcdf(t2) )
end

# cdf

function cdf(d::Exgaussian, x::Real)
    μ, σ, rate = d.μ, d.σ, d.rate
    t3a = μ / σ + σ*rate;
    t4a = (σ*rate)^2 / 2;
    t1 = normcdf((x-μ)/σ)
    t3 = normcdf(x/σ-t3a)
    t4 = rate*(μ-x) + t4a
    t2 = exp(t4)
    return t1 - t2.*t3
end

function mgf(d::Exgaussian, s::Real)
    return exp(s*d.μ + ((s*d.σ)^2)/2) / (1 - s/d.rate)
end

#### Sampling

rand(rng::AbstractRNG, d::Exgaussian{T}) where {T} = d.μ + d.σ * randn(rng, float(T)) + d.τ*randexp(rng)
