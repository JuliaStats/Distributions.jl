"""
    Erlang(α,θ)

The *Erlang distribution* is a special case of a [`Gamma`](@ref) distribution with integer shape parameter.

```julia
Erlang()       # Erlang distribution with unit shape and unit scale, i.e. Erlang(1, 1)
Erlang(a)      # Erlang distribution with shape parameter a and unit scale, i.e. Erlang(a, 1)
Erlang(a, s)   # Erlang distribution with shape parameter a and scale s
```

External links

* [Erlang distribution on Wikipedia](http://en.wikipedia.org/wiki/Erlang_distribution)

"""
struct Erlang{T<:Real} <: ContinuousUnivariateDistribution
    α::Int
    θ::T
    Erlang{T}(α::Int, θ::T) where {T} = new{T}(α, θ)
end

function Erlang(α::Real, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Erlang, isinteger(α) && α >= zero(α))
    return Erlang{T}(α, θ)
end

function Erlang(α::Integer, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Erlang, α >= zero(α))
    return Erlang{T}(α, θ)
end

function Erlang(α::Integer, θ::Integer)
    θf = float(θ)
    return Erlang{typeof(θf)}(α, θf)
end

Erlang(α::Integer) = Erlang(α, 1.0)
Erlang() = Erlang(1, 1.0, check_args=false)

@distr_support Erlang 0.0 Inf

#### Conversions
function convert(::Type{Erlang{T}}, α::Integer, θ::S) where {T <: Real, S <: Real}
    Erlang(α, T(θ), check_args=false)
end
function convert(::Type{Erlang{T}}, d::Erlang{S}) where {T <: Real, S <: Real}
    Erlang(d.α, T(d.θ), check_args=false)
end

#### Parameters

shape(d::Erlang) = d.α
scale(d::Erlang) = d.θ
rate(d::Erlang) = inv(d.θ)
params(d::Erlang) = (d.α, d.θ)
@inline partype(d::Erlang{T}) where {T<:Real} = T

#### Statistics

mean(d::Erlang) = d.α * d.θ
var(d::Erlang) = d.α * d.θ^2
skewness(d::Erlang) = 2 / sqrt(d.α)
kurtosis(d::Erlang) = 6 / d.α

function mode(d::Erlang)
    (α, θ) = params(d)
    α >= 1 ? θ * (α - 1) : error("Erlang has no mode when α < 1")
end

function entropy(d::Erlang)
    (α, θ) = params(d)
    α + loggamma(α) + (1 - α) * digamma(α) + log(θ)
end

mgf(d::Erlang, t::Real) = (1 - t * d.θ)^(-d.α)
cf(d::Erlang, t::Real)  = (1 - im * t * d.θ)^(-d.α)


#### Evaluation & Sampling

@_delegate_statsfuns Erlang gamma α θ

rand(rng, ::AbstractRNG, d::Erlang) = rand(rng, Gamma(Float64(d.α), d.θ))
sampler(d::Erlang) = Gamma(Float64(d.α), d.θ)
