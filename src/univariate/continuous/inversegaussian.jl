"""
    InverseGaussian <: ContinuousUnivariateDistribution

The *inverse Gaussian* probability distribution

# Constructors

    InverseGaussian(μ|mu|mean=1, λ|lambda|shape=1)

Construct an `InverseGaussian` distribution object with mean `μ` and shape `λ`.

    InverseGaussian(μ|mu|mean=, std=)
    InverseGaussian(μ|mu|mean=, var=)

Construct an `InverseGaussian` distribution object matching the relevant moments.

# Details

The inverse Gaussian distribution has probability density function

```math
f(x; \\mu, \\lambda) = \\sqrt{\\frac{\\lambda}{2\\pi x^3}}
\\exp\\!\\left(\\frac{-\\lambda(x-\\mu)^2}{2\\mu^2x}\\right), \\quad x > 0
```

# Examples

```julia
InverseGaussian()
InverseGaussian(mean=2, std=3)
```

# External links

* [Inverse Gaussian distribution on Wikipedia](http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)

"""
struct InverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    λ::T

    function InverseGaussian{T}(μ::T, λ::T) where T
        @check_args(InverseGaussian, μ > zero(μ) && λ > zero(λ))
        new{T}(μ, λ)
    end
end

InverseGaussian(μ::T, λ::T) where {T<:Real} = InverseGaussian{T}(μ, λ)
InverseGaussian(μ::Real, λ::Real) = InverseGaussian(promote(μ, λ)...)
InverseGaussian(μ::Integer, λ::Integer) = InverseGaussian(float(μ), float(λ))

@kwdispatch (::Type{D})(;mu=>μ, mean=>μ, lambda=>λ, shape=>λ) where {D<:InverseGaussian} begin
    () -> D(1,1)
    (μ) -> D(μ,1)
    (λ) -> D(1,λ)
    (μ,λ) -> D(μ,λ)

    (μ,var) = D(μ, μ^3/var)
    (μ,std) = D(μ, μ^3/std^2)
end


@distr_support InverseGaussian 0.0 Inf

#### Conversions

function convert(::Type{InverseGaussian{T}}, μ::S, λ::S) where {T <: Real, S <: Real}
    InverseGaussian(T(μ), T(λ))
end
function convert(::Type{InverseGaussian{T}}, d::InverseGaussian{S}) where {T <: Real, S <: Real}
    InverseGaussian(T(d.μ), T(d.λ))
end

#### Parameters

shape(d::InverseGaussian) = d.λ
params(d::InverseGaussian) = (d.μ, d.λ)
@inline partype(d::InverseGaussian{T}) where {T<:Real} = T


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
rand(d::InverseGaussian) = rand(GLOBAL_RNG, d)
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
