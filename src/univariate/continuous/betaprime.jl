"""
    BetaPrime(α,β)

The *Beta prime distribution* has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)}
x^{\\alpha - 1} (1 + x)^{- (\\alpha + \\beta)}, \\quad x > 0
```


The Beta prime distribution is related to the [`Beta`](@ref) distribution via the
relation ship that if ``X \\sim \\operatorname{Beta}(\\alpha, \\beta)`` then ``\\frac{X}{1 - X}
\\sim \\operatorname{BetaPrime}(\\alpha, \\beta)``

```julia
BetaPrime()        # equivalent to BetaPrime(1, 1)
BetaPrime(a)       # equivalent to BetaPrime(a, a)
BetaPrime(a, b)    # Beta prime distribution with shape parameters a and b

params(d)          # Get the parameters, i.e. (a, b)
```

External links

* [Beta prime distribution on Wikipedia](http://en.wikipedia.org/wiki/Beta_prime_distribution)

"""
struct BetaPrime{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    BetaPrime{T}(α::T, β::T) where {T} = new{T}(α, β)
end

function BetaPrime(α::T, β::T; check_args=true) where {T<:Real}
    check_args && @check_args(BetaPrime, α > zero(α) && β > zero(β))
    return BetaPrime{T}(α, β)
end

BetaPrime(α::Real, β::Real) = BetaPrime(promote(α, β)...)
BetaPrime(α::Integer, β::Integer) = BetaPrime(float(α), float(β))
BetaPrime(α::Real) = BetaPrime(α, α)
BetaPrime() = BetaPrime(1.0, 1.0, check_args=false)

@distr_support BetaPrime 0.0 Inf

#### Conversions
function convert(::Type{BetaPrime{T}}, α::Real, β::Real) where T<:Real
    BetaPrime(T(α), T(β))
end
function convert(::Type{BetaPrime{T}}, d::BetaPrime{S}) where {T <: Real, S <: Real}
    BetaPrime(T(d.α), T(d.β), check_args=false)
end

#### Parameters

params(d::BetaPrime) = (d.α, d.β)
@inline partype(d::BetaPrime{T}) where {T<:Real} = T

#### Statistics

function mean(d::BetaPrime{T}) where T<:Real
    ((α, β) = params(d); β > 1 ? α / (β - 1) : T(NaN))
end

function mode(d::BetaPrime{T}) where T<:Real
    ((α, β) = params(d); α > 1 ? (α - 1) / (β + 1) : zero(T))
end

function var(d::BetaPrime{T}) where T<:Real
    (α, β) = params(d)
    β > 2 ? α * (α + β - 1) / ((β - 2) * (β - 1)^2) : T(NaN)
end

function skewness(d::BetaPrime{T}) where T<:Real
    (α, β) = params(d)
    if β > 3
        s = α + β - 1
        2(α + s) / (β - 3) * sqrt((β - 2) / (α * s))
    else
        return T(NaN)
    end
end


#### Evaluation

function logpdf(d::BetaPrime{T}, x::Real) where T<:Real
    (α, β) = params(d)
    if x < 0
        T(-Inf)
    else
        (α - 1) * log(x) - (α + β) * log1p(x) - logbeta(α, β)
    end
end

pdf(d::BetaPrime, x::Real) = exp(logpdf(d, x))

cdf(d::BetaPrime{T}, x::Real) where {T<:Real} = x <= 0 ? zero(T) : betacdf(d.α, d.β, x / (1 + x))
ccdf(d::BetaPrime{T}, x::Real) where {T<:Real} = x <= 0 ? one(T) : betaccdf(d.α, d.β, x / (1 + x))
logcdf(d::BetaPrime{T}, x::Real) where {T<:Real} =  x <= 0 ? T(-Inf) : betalogcdf(d.α, d.β, x / (1 + x))
logccdf(d::BetaPrime{T}, x::Real) where {T<:Real} =  x <= 0 ? zero(T) : betalogccdf(d.α, d.β, x / (1 + x))

quantile(d::BetaPrime, p::Real) = (x = betainvcdf(d.α, d.β, p); x / (1 - x))
cquantile(d::BetaPrime, p::Real) = (x = betainvccdf(d.α, d.β, p); x / (1 - x))
invlogcdf(d::BetaPrime, p::Real) = (x = betainvlogcdf(d.α, d.β, p); x / (1 - x))
invlogccdf(d::BetaPrime, p::Real) = (x = betainvlogccdf(d.α, d.β, p); x / (1 - x))


#### Sampling

function rand(rng::AbstractRNG, d::BetaPrime)
    (α, β) = params(d)
    rand(rng, Gamma(α)) / rand(rng, Gamma(β))
end
