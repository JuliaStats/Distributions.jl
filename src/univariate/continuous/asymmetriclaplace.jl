"""
    AsymmetricLaplace(μ,λ,κ)

The *Asymmetric Laplace distribution* with location `μ` scale `λ` and asymmetry `κ` has probability density function

```math
f(x; \\mu, \\lambda, \\kappa) = \\left( \\frac{\\lambda}{\\kappa + 1/\\kappa} \\right) \\exp - (x - \\mu) \\lambda \\sgn(x - \\mu) \\kappa^\\sgn(x - \\mu)
```

```julia
AsymmetricLaplace()          # Asymmetric Laplace distribution with zero location, unit scale and asymmetry 1, i.e. AsymmetricLaplace(0, 1, 1)
AsymmetricLaplace(μ)         # Asymmetric Laplace distribution with location μ, unit scale and asymmetry 1, i.e. AsymmetricLaplace(μ, 1, 1)
AsymmetricLaplace(μ, λ)      # Asymmetric Laplace distribution with location μ, scale λ and asymmetry 1
AsymmetricLaplace(μ, λ, κ)   # Asymmetric Laplace distribution with location μ, scale λ and asymmetry κ

params(d)                    # Get the parameters, i.e., (μ, λ, κ)
location(d)                  # Get the location parameter, i.e. μ
scale(d)                     # Get the scale parameter, i.e. λ
asymmetry(d)                 # Get the asymmetry parameter, i.e. κ
```

External links

* [Asymmetric Laplace distribution on Wikipedia](http://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution)

"""
struct AsymmetricLaplace{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    λ::T
    κ::T
    AsymmetricLaplace{T}(µ::T, λ::T, κ::T) where {T} = new{T}(µ, λ, κ)
end

function AsymmetricLaplace(μ::T, λ::T, κ::T; check_args=true) where {T <: Real}
    check_args && @check_args(AsymmetricLaplace, λ > zero(λ) && κ > zero(κ))
    return AsymmetricLaplace{T}(μ, λ, κ)
end

AsymmetricLaplace(μ::Real, λ::Real, κ::Real) = AsymmetricLaplace(promote(μ, λ, κ)...)
AsymmetricLaplace(μ::Integer, λ::Integer, κ::Integer) = AsymmetricLaplace(float(μ), float(λ), float(κ))
AsymmetricLaplace(μ::T) where {T <: Real} = AsymmetricLaplace(μ, one(T), one(T))
AsymmetricLaplace(μ::T, λ::T) where {T <: Real} = AsymmetricLaplace(μ, λ, one(T))
AsymmetricLaplace() = AsymmetricLaplace(0.0, 1.0, 1.0, check_args=false)

@distr_support AsymmetricLaplace -Inf Inf

#### Conversions
function convert(::Type{AsymmetricLaplace{T}}, μ::S, λ::S, κ::S) where {T <: Real, S <: Real}
    AsymmetricLaplace(T(μ), T(λ), T(κ))
end
function convert(::Type{AsymmetricLaplace{T}}, d::AsymmetricLaplace{S}) where {T <: Real, S <: Real}
    AsymmetricLaplace(T(d.μ), T(d.λ), T(d.κ), check_args=false)
end

#### Parameters

location(d::AsymmetricLaplace) = d.μ
scale(d::AsymmetricLaplace) = d.λ
asymmetry(d::AsymmetricLaplace) = d.κ
params(d::AsymmetricLaplace) = (d.μ, d.λ, d.κ)
@inline partype(d::AsymmetricLaplace{T}) where {T<:Real} = T


#### Statistics

mean(d::AsymmetricLaplace) = d.μ + (1 - d.κ^2) / (d.λ * d.κ)
function median(d::AsymmetricLaplace)
    μ, λ, κ = params(d)
    if κ > 1
        μ + κ/λ * log((1 + κ^2)/(2κ^2))
    else
        μ - 1/(κ * λ) * log((1 + κ^2)/2)
    end
end
mode(d::AsymmetricLaplace) = d.μ

var(d::AsymmetricLaplace) = (1 + d.κ^4) / (d.λ^2 * d.κ^2)
std(d::AsymmetricLaplace) = sqrt(1 + d.κ^4) / (d.λ * d.κ)
skewness(d::AsymmetricLaplace) = 2(1 - d.κ^6) / (d.κ^4 + 1)^(3/2)
kurtosis(d::AsymmetricLaplace) = 6(1 + d.κ^8) / (1 + d.κ^4)^2

entropy(d::AsymmetricLaplace) = log(ℯ * (1 + d.κ) / (d.λ * d.κ))

#### Evaluations

function pdf(d::AsymmetricLaplace, x::Real)
    μ, λ, κ = params(d)
    if x ≤ μ
        exp((x - μ) / (λ * κ)) / λ / (κ + 1 / κ)
    else
        exp(- κ / λ * (x - μ)) / λ / (κ + 1 / κ)
    end
end
function logpdf(d::AsymmetricLaplace, x::Real)
    μ, λ, κ = params(d)
    if x ≤ μ
        (x - μ) / (λ * κ) - log(λ) - log(κ + 1 / κ)
    else
        - κ * (x - μ) / λ - log(λ) - log(κ + 1 / κ)
    end
end

function cdf(d::AsymmetricLaplace, x::Real)
    μ, λ, κ = params(d)
    if x ≤ μ
        κ^2 / (1 + κ^2) * exp((x - μ) / (λ * κ))
    else
        1 - 1 / (1 + κ^2) * exp(- κ / λ * (x - μ))
    end
end
ccdf(d::AsymmetricLaplace, x::Real) = 1 - cdf(d, x)
function logcdf(d::AsymmetricLaplace, x::Real)
    μ, λ, κ = params(d)
    if x ≤ μ
        2 * log(κ) - log(1 + κ^2) + (x - μ) / (λ * κ)
    else
        log(1 - 1 / (1 + κ^2) * exp(- κ / λ * (x - μ)))
    end
end
function logccdf(d::AsymmetricLaplace, x::Real)
    μ, λ, κ = params(d)
    if x ≤ μ
        log(1 - κ^2 / (1 + κ^2) * exp((x - μ) / (λ * κ)))
    else
        -(κ * (x - μ) / λ  + log(1 + κ^2))
    end
end

function quantile(d::AsymmetricLaplace, p::Real)
    μ, λ, κ = params(d)
    if p ≤ cdf(d, d.μ)
        μ + log((1 + κ^2) / κ^2 * p) * (λ * κ)
    else
        μ - λ / κ * log((1 - p) * (1 + κ^2))
    end
end
cquantile(d::AsymmetricLaplace, p::Real) = 1 - quantile(d, p)
function invlogcdf(d::AsymmetricLaplace, lp::Real)
    μ, λ, κ = params(d)
    if lp ≤ 0
        println("branch less than")
        μ - (λ / κ) * (log(1 - exp(lp)) + log(1 + κ^2))
    else
        println("branch larger than")
        μ + λ * κ * (log(lp) + log(1 + κ^2) - 2 * log(κ))
    end
end
function invlogccdf(d::AsymmetricLaplace, lp::Real)
    μ, λ, κ = params(d)
    if lp ≤ log(cdf(d, μ))
        μ - (λ / κ) * (lp + log(1 + κ^2))
    else
        μ + λ * κ * log((1 + κ^2) * (1 - exp(lp)) / κ^2)
    end
end

function gradlogpdf(d::AsymmetricLaplace, x::Real)
    x == d.μ && error("Gradient is undefined at the location point")
    x > d.μ ? - d.κ / d.λ : 1 / (d.λ * d.κ)
end

function mgf(d::AsymmetricLaplace, t::Real)
    exp(t * d.μ) / ((1 + t * d.κ / d.λ) * (1 - t / (d.κ * d.λ)))
end
function cf(d::AsymmetricLaplace, t::Real)
    cis(t * d.μ) / ((1 + t * d.κ * im / d.λ) * (1 - t * im / (d.κ * d.λ)))
end

#### Sampling

rand(rng::AbstractRNG, d::AsymmetricLaplace) = quantile(d, rand(rng))
