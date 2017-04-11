doc"""
    Levy(μ, σ)

The *Lévy distribution* with location `μ` and scale `σ` has probability density function

$f(x; \mu, \sigma) = \sqrt{\frac{\sigma}{2 \pi (x - \mu)^3}}
\exp \left( - \frac{\sigma}{2 (x - \mu)} \right), \quad x > \mu$

```julia
Levy()         # Levy distribution with zero location and unit scale, i.e. Levy(0, 1)
Levy(u)        # Levy distribution with location u and unit scale, i.e. Levy(u, 1)
Levy(u, c)     # Levy distribution with location u ans scale c

params(d)      # Get the parameters, i.e. (u, c)
location(d)    # Get the location parameter, i.e. u
```

External links

* [Lévy distribution on Wikipedia](http://en.wikipedia.org/wiki/Lévy_distribution)
"""
immutable Levy{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    (::Type{Levy{T}}){T}(μ::T, σ::T) = (@check_args(Levy, σ > zero(σ)); new{T}(μ, σ))
end

Levy{T<:Real}(μ::T, σ::T) = Levy{T}(μ, σ)
Levy(μ::Real, σ::Real) = Levy(promote(μ, σ)...)
Levy(μ::Integer, σ::Integer) = Levy(Float64(μ), Float64(σ))
Levy(μ::Real) = Levy(μ, 1.0)
Levy() = Levy(0.0, 1.0)

@distr_support Levy d.μ Inf

#### Conversions

convert{T <: Real, S <: Real}(::Type{Levy{T}}, μ::S, σ::S) = Levy(T(μ), T(σ))
convert{T <: Real, S <: Real}(::Type{Levy{T}}, d::Levy{S}) = Levy(T(d.μ), T(d.σ))

#### Parameters

location(d::Levy) = d.μ
params(d::Levy) = (d.μ, d.σ)
@inline partype{T<:Real}(d::Levy{T}) = T


#### Statistics

mean{T<:Real}(d::Levy{T}) = T(Inf)
var{T<:Real}(d::Levy{T}) = T(Inf)
skewness{T<:Real}(d::Levy{T}) = T(NaN)
kurtosis{T<:Real}(d::Levy{T}) = T(NaN)

mode(d::Levy) = d.σ / 3 + d.μ

entropy(d::Levy) = (1 - 3digamma(1) + log(16 * d.σ^2 * π)) / 2

median{T<:Real}(d::Levy{T}) = d.μ + d.σ / (2 * T(erfcinv(0.5))^2)


#### Evaluation

function pdf(d::Levy, x::Real)
    μ, σ = params(d)
    z = x - μ
    (sqrt(σ) / sqrt2π) * exp((-σ) / (2z)) / z^(3//2)
end

function logpdf(d::Levy, x::Real)
    μ, σ = params(d)
    z = x - μ
    (log(σ) - log2π - σ / z - 3log(z))/2
end

cdf(d::Levy, x::Real) = erfc(sqrt(d.σ / (2(x - d.μ))))
ccdf(d::Levy, x::Real) = erf(sqrt(d.σ / (2(x - d.μ))))

quantile(d::Levy, p::Real) = d.μ + d.σ / (2*erfcinv(p)^2)
cquantile(d::Levy, p::Real) = d.μ + d.σ / (2*erfinv(p)^2)

mgf{T<:Real}(d::Levy{T}, t::Real) = t == zero(t) ? one(T) : T(NaN)

function cf(d::Levy, t::Real)
    μ, σ = params(d)
    exp(im * μ * t - sqrt(-2im * σ * t))
end


#### Sampling

rand(d::Levy) = d.μ + d.σ / randn()^2
