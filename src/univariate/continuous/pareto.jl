"""
    Pareto(α,θ)

The *Pareto distribution* with shape `α` and scale `θ` has probability density function

```math
f(x; \\alpha, \\theta) = \\frac{\\alpha \\theta^\\alpha}{x^{\\alpha + 1}}, \\quad x \\ge \\theta
```

```julia
Pareto()            # Pareto distribution with unit shape and unit scale, i.e. Pareto(1, 1)
Pareto(a)           # Pareto distribution with shape a and unit scale, i.e. Pareto(a, 1)
Pareto(a, b)        # Pareto distribution with shape a and scale b

params(d)        # Get the parameters, i.e. (a, b)
shape(d)         # Get the shape parameter, i.e. a
scale(d)         # Get the scale parameter, i.e. b
```

External links
 * [Pareto distribution on Wikipedia](http://en.wikipedia.org/wiki/Pareto_distribution)

"""
struct Pareto{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T
    Pareto{T}(α::T, θ::T) where {T} = new{T}(α, θ)
end

function Pareto(α::T, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Pareto, α > zero(α) && θ > zero(θ))
    return Pareto{T}(α, θ)
end

Pareto(α::Real, θ::Real) = Pareto(promote(α, θ)...)
Pareto(α::Integer, θ::Integer) = Pareto(float(α), float(θ))
Pareto(α::T) where {T <: Real} = Pareto(α, one(T))
Pareto() = Pareto(1.0, 1.0, check_args=false)

@distr_support Pareto d.θ Inf

#### Conversions
convert(::Type{Pareto{T}}, α::Real, θ::Real) where {T<:Real} = Pareto(T(α), T(θ))
convert(::Type{Pareto{T}}, d::Pareto{S}) where {T <: Real, S <: Real} = Pareto(T(d.α), T(d.θ), check_args=false)

#### Parameters

shape(d::Pareto) = d.α
scale(d::Pareto) = d.θ

params(d::Pareto) = (d.α, d.θ)
@inline partype(d::Pareto{T}) where {T<:Real} = T


#### Statistics

function mean(d::Pareto{T}) where T<:Real
    (α, θ) = params(d)
    α > 1 ? α * θ / (α - 1) : T(Inf)
end
median(d::Pareto) = ((α, θ) = params(d); θ * 2^(1/α))
mode(d::Pareto) = d.θ

function var(d::Pareto{T}) where T<:Real
    (α, θ) = params(d)
    α > 2 ? (θ^2 * α) / ((α - 1)^2 * (α - 2)) : T(Inf)
end

function skewness(d::Pareto{T}) where T<:Real
    α = shape(d)
    α > 3 ? ((2(1 + α)) / (α - 3)) * sqrt((α - 2) / α) : T(NaN)
end

function kurtosis(d::Pareto{T}) where T<:Real
    α = shape(d)
    α > 4 ? (6(α^3 + α^2 - 6α - 2)) / (α * (α - 3) * (α - 4)) : T(NaN)
end

entropy(d::Pareto) = ((α, θ) = params(d); log(θ / α) + 1 / α + 1)


#### Evaluation

function pdf(d::Pareto{T}, x::Real) where T<:Real
    (α, θ) = params(d)
    x >= θ ? α * (θ / x)^α * (1/x) : zero(T)
end

function logpdf(d::Pareto{T}, x::Real) where T<:Real
    (α, θ) = params(d)
    x >= θ ? log(α) + α * log(θ) - (α + 1) * log(x) : -T(Inf)
end

function ccdf(d::Pareto{T}, x::Real) where T<:Real
    (α, θ) = params(d)
    x >= θ ? (θ / x)^α : one(T)
end

cdf(d::Pareto, x::Real) = 1 - ccdf(d, x)

function logccdf(d::Pareto{T}, x::Real) where T<:Real
    (α, θ) = params(d)
    x >= θ ? α * log(θ / x) : zero(T)
end

logcdf(d::Pareto, x::Real) = log1p(-ccdf(d, x))

cquantile(d::Pareto, p::Real) = d.θ / p^(1 / d.α)
quantile(d::Pareto, p::Real) = cquantile(d, 1 - p)


#### Sampling

rand(rng::AbstractRNG, d::Pareto) = d.θ * exp(randexp(rng) / d.α)

## Fitting

function fit_mle(::Type{<:Pareto}, x::AbstractArray{T}) where T<:Real
    # Based on
    # https://en.wikipedia.org/wiki/Pareto_distribution#Parameter_estimation

    θ = minimum(x)

    n = length(x)
    lθ = log(θ)
    temp1 = zero(T)
    for i=1:n
        temp1 += log(x[i]) - lθ
    end
    α = n/temp1

    return Pareto(α, θ)
end
