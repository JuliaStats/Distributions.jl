"""
    Levy(μ, σ)

The *Lévy distribution* with location `μ` and scale `σ` has probability density function

```math
f(x; \\mu, \\sigma) = \\sqrt{\\frac{\\sigma}{2 \\pi (x - \\mu)^3}}
\\exp \\left( - \\frac{\\sigma}{2 (x - \\mu)} \\right), \\quad x > \\mu
```

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
struct Levy{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
end

function Levy(μ::T, σ::T; check_args=true) where {T}
    check_args && @check_args(Levy, σ > zero(σ))
    return Levy{T}(μ, σ)
end

Levy(μ::Real, σ::Real) = Levy(promote(μ, σ)...)
Levy(μ::Integer, σ::Integer) = Levy(float(μ), float(σ))
Levy(μ::T) where {T <: Real} = Levy(μ, one(T))
Levy() = Levy(0.0, 1.0, check_args=false)

@distr_support Levy d.μ Inf

#### Conversions

convert(::Type{Levy{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = Levy(T(μ), T(σ))
convert(::Type{Levy{T}}, d::Levy{S}) where {T <: Real, S <: Real} = Levy(T(d.μ), T(d.σ), check_args=false)

#### Parameters

location(d::Levy) = d.μ
params(d::Levy) = (d.μ, d.σ)
partype(::Levy{T}) where {T} = T


#### Statistics

mean(d::Levy{T}) where {T<:Real} = T(Inf)
var(d::Levy{T}) where {T<:Real} = T(Inf)
skewness(d::Levy{T}) where {T<:Real} = T(NaN)
kurtosis(d::Levy{T}) where {T<:Real} = T(NaN)

mode(d::Levy) = d.σ / 3 + d.μ

entropy(d::Levy) = (1 - 3digamma(1) + log(16 * d.σ^2 * π)) / 2

median(d::Levy{T}) where {T<:Real} = d.μ + d.σ / (2 * T(erfcinv(0.5))^2)


#### Evaluation

function pdf(d::Levy{T}, x::Real) where T<:Real
    μ, σ = params(d)
    if x <= μ
        return zero(T)
    end
    z = x - μ
    (sqrt(σ) / sqrt2π) * exp((-σ) / (2z)) / z^(3//2)
end

function logpdf(d::Levy{T}, x::Real) where T<:Real
    μ, σ = params(d)
    if x <= μ
        return T(-Inf)
    end
    z = x - μ
    (log(σ) - log2π - σ / z - 3log(z))/2
end

cdf(d::Levy{T}, x::Real) where {T<:Real} = x <= d.μ ? zero(T) : erfc(sqrt(d.σ / (2(x - d.μ))))
ccdf(d::Levy{T}, x::Real) where {T<:Real} =  x <= d.μ ? one(T) : erf(sqrt(d.σ / (2(x - d.μ))))

quantile(d::Levy, p::Real) = d.μ + d.σ / (2*erfcinv(p)^2)
cquantile(d::Levy, p::Real) = d.μ + d.σ / (2*erfinv(p)^2)

mgf(d::Levy{T}, t::Real) where {T<:Real} = t == zero(t) ? one(T) : T(NaN)

function cf(d::Levy, t::Real)
    μ, σ = params(d)
    exp(im * μ * t - sqrt(-2im * σ * t))
end


#### Sampling

rand(rng::AbstractRNG, d::Levy) = d.μ + d.σ / randn(rng)^2
