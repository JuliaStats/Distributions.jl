"""
    InverseGaussian(μ,λ)

The *inverse Gaussian distribution* with mean `μ` and shape `λ` has probability density function

```math
f(x; \\mu, \\lambda) = \\sqrt{\\frac{\\lambda}{2\\pi x^3}}
\\exp\\!\\left(\\frac{-\\lambda(x-\\mu)^2}{2\\mu^2x}\\right), \\quad x > 0
```

```julia
InverseGaussian()              # Inverse Gaussian distribution with unit mean and unit shape, i.e. InverseGaussian(1, 1)
InverseGaussian(mu),           # Inverse Gaussian distribution with mean mu and unit shape, i.e. InverseGaussian(u, 1)
InverseGaussian(mu, lambda)    # Inverse Gaussian distribution with mean mu and shape lambda

params(d)           # Get the parameters, i.e. (mu, lambda)
mean(d)             # Get the mean parameter, i.e. mu
shape(d)            # Get the shape parameter, i.e. lambda
```

External links

* [Inverse Gaussian distribution on Wikipedia](http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)

"""
struct InverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    λ::T
end

function InverseGaussian(μ::T, λ::T; check_args=true) where {T}
    check_args && @check_args(InverseGaussian, μ > zero(μ) && λ > zero(λ))
    return InverseGaussian{T}(μ, λ)
end

InverseGaussian(μ::Real, λ::Real) = InverseGaussian(promote(μ, λ)...)
InverseGaussian(μ::Integer, λ::Integer) = InverseGaussian(float(μ), float(λ))
InverseGaussian(μ::T) where {T <: Real} = InverseGaussian(μ, one(T))
InverseGaussian() = InverseGaussian(1.0, 1.0, check_args=false)

@distr_support InverseGaussian 0.0 Inf

#### Conversions

function convert(::Type{InverseGaussian{T}}, μ::S, λ::S) where {T <: Real, S <: Real}
    InverseGaussian(T(μ), T(λ))
end
function convert(::Type{InverseGaussian{T}}, d::InverseGaussian{S}) where {T <: Real, S <: Real}
    InverseGaussian(T(d.μ), T(d.λ), check_args=false)
end

#### Parameters

shape(d::InverseGaussian) = d.λ
params(d::InverseGaussian) = (d.μ, d.λ)
partype(::InverseGaussian{T}) where {T} = T

#### Statistics

mean(d::InverseGaussian) = d.μ

var(d::InverseGaussian) = d.μ^3 / d.λ

skewness(d::InverseGaussian) = 3sqrt(d.μ / d.λ)

kurtosis(d::InverseGaussian) = 15d.μ / d.λ

function mode(d::InverseGaussian)
    μ, λ = params(d)
    r = μ / λ
    μ * (sqrt(1 + (3r/2)^2) - (3r/2))
end


#### Evaluation

function pdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        return sqrt(λ / (twoπ * x^3)) * exp(-λ * (x - μ)^2 / (2μ^2 * x))
    else
        return zero(T)
    end
end

function logpdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        return (log(λ) - (log2π + 3log(x)) - λ * (x - μ)^2 / (μ^2 * x))/2
    else
        return -T(Inf)
    end
end

function cdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        return normcdf(u * (v - 1)) + exp(2λ / μ) * normcdf(-u * (v + 1))
    else
        return zero(T)
    end
end

function ccdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        normccdf(u * (v - 1)) - exp(2λ / μ) * normcdf(-u * (v + 1))
    else
        return one(T)
    end
end

function logcdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        a = normlogcdf(u * (v -1))
        b = 2λ / μ + normlogcdf(-u * (v + 1))
        a + log1pexp(b - a)
    else
        return -T(Inf)
    end
end

function logccdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        a = normlogccdf(u * (v - 1))
        b = 2λ / μ + normlogcdf(-u * (v + 1))
        a + log1mexp(b - a)
    else
        return zero(T)
    end
end

@quantile_newton InverseGaussian

#### Sampling

# rand method from:
#   John R. Michael, William R. Schucany and Roy W. Haas (1976)
#   Generating Random Variates Using Transformations with Multiple Roots
#   The American Statistician , Vol. 30, No. 2, pp. 88-90
function rand(rng::AbstractRNG, d::InverseGaussian)
    μ, λ = params(d)
    z = randn(rng)
    v = z * z
    w = μ * v
    x1 = μ + μ / (2λ) * (w - sqrt(w * (4λ + w)))
    p1 = μ / (μ + x1)
    u = rand(rng)
    u >= p1 ? μ^2 / x1 : x1
end

#### Fit model

"""
Sufficient statistics for `InverseGaussian`, containing the weighted
sum of observations, the weighted sum of inverse points and sum of weights.
"""
struct InverseGaussianStats <: SufficientStats
    sx::Float64      # (weighted) sum of x
    sinvx::Float64   # (weighted) sum of 1/x
    sw::Float64      # sum of sample weight
end

function suffstats(::Type{<:InverseGaussian}, x::AbstractVector{<:Real})
    sx = sum(x)
    sinvx = sum(inv, x)
    InverseGaussianStats(sx, sinvx, length(x))
end

function suffstats(::Type{<:InverseGaussian}, x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    n = length(x)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
    T = promote_type(eltype(x), eltype(w))
    sx = zero(T)
    sinvx = zero(T)
    sw = zero(T)
    @inbounds @simd for i in eachindex(x)
        sx += w[i]*x[i]
        sinvx += w[i]/x[i]
        sw += w[i]
    end
    InverseGaussianStats(sx, sinvx, sw)
end

function fit_mle(::Type{<:InverseGaussian}, ss::InverseGaussianStats)
    mu = ss.sx / ss.sw
    invlambda = ss.sinvx / ss.sw  -  inv(mu)
    InverseGaussian(mu, inv(invlambda))
end
