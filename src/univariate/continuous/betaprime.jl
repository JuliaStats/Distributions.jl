doc"""
    BetaPrime(α,β)

The *Beta prime distribution* has probability density function

$f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}
x^{\alpha - 1} (1 + x)^{- (\alpha + \beta)}, \quad x > 0$


The Beta prime distribution is related to the [`Beta`](@ref) distribution via the
relation ship that if $X \sim \operatorname{Beta}(\alpha, \beta)$ then $\frac{X}{1 - X}
\sim \operatorname{BetaPrime}(\alpha, \beta)$

```julia
BetaPrime()        # equivalent to BetaPrime(1, 1)
BetaPrime(a)       # equivalent to BetaPrime(a, a)
BetaPrime(a, b)    # Beta prime distribution with shape parameters a and b

params(d)          # Get the parameters, i.e. (a, b)
```

External links

* [Beta prime distribution on Wikipedia](http://en.wikipedia.org/wiki/Beta_prime_distribution)

    """

immutable BetaPrime{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T

    function (::Type{BetaPrime{T}}){T}(α::T, β::T)
        @check_args(BetaPrime, α > zero(α) && β > zero(β))
        new{T}(α, β)
    end
end

BetaPrime{T<:Real}(α::T, β::T) = BetaPrime{T}(α, β)
BetaPrime(α::Real, β::Real) = BetaPrime(promote(α, β)...)
BetaPrime(α::Integer, β::Integer) = BetaPrime(Float64(α), Float64(β))
BetaPrime(α::Real) = BetaPrime(α, α)
BetaPrime() = BetaPrime(1.0, 1.0)

@distr_support BetaPrime 0.0 Inf

#### Conversions
function convert{T<:Real}(::Type{BetaPrime{T}}, α::Real, β::Real)
    BetaPrime(T(α), T(β))
end
function convert{T <: Real, S <: Real}(::Type{BetaPrime{T}}, d::BetaPrime{S})
    BetaPrime(T(d.α), T(d.β))
end

#### Parameters

params(d::BetaPrime) = (d.α, d.β)
@inline partype{T<:Real}(d::BetaPrime{T}) = T

#### Statistics

function mean{T<:Real}(d::BetaPrime{T})
    ((α, β) = params(d); β > 1 ? α / (β - 1) : T(NaN))
end

function mode{T<:Real}(d::BetaPrime{T})
    ((α, β) = params(d); α > 1 ? (α - 1) / (β + 1) : zero(T))
end

function var{T<:Real}(d::BetaPrime{T})
    (α, β) = params(d)
    β > 2 ? α * (α + β - 1) / ((β - 2) * (β - 1)^2) : T(NaN)
end

function skewness{T<:Real}(d::BetaPrime{T})
    (α, β) = params(d)
    if β > 3
        s = α + β - 1
        2(α + s) / (β - 3) * sqrt((β - 2) / (α * s))
    else
        return T(NaN)
    end
end


#### Evaluation

function logpdf{T<:Real}(d::BetaPrime{T}, x::Real)
    (α, β) = params(d)
    if x < 0
        T(-Inf)
    else
        (α - 1) * log(x) - (α + β) * log1p(x) - lbeta(α, β)
    end
end

pdf(d::BetaPrime, x::Real) = exp(logpdf(d, x))

cdf{T<:Real}(d::BetaPrime{T}, x::Real) = x <= 0 ? zero(T) : betacdf(d.α, d.β, x / (1 + x))
ccdf{T<:Real}(d::BetaPrime{T}, x::Real) = x <= 0 ? one(T) : betaccdf(d.α, d.β, x / (1 + x))
logcdf{T<:Real}(d::BetaPrime{T}, x::Real) =  x <= 0 ? T(-Inf) : betalogcdf(d.α, d.β, x / (1 + x))
logccdf{T<:Real}(d::BetaPrime{T}, x::Real) =  x <= 0 ? zero(T) : betalogccdf(d.α, d.β, x / (1 + x))

quantile(d::BetaPrime, p::Real) = (x = betainvcdf(d.α, d.β, p); x / (1 - x))
cquantile(d::BetaPrime, p::Real) = (x = betainvccdf(d.α, d.β, p); x / (1 - x))
invlogcdf(d::BetaPrime, p::Real) = (x = betainvlogcdf(d.α, d.β, p); x / (1 - x))
invlogccdf(d::BetaPrime, p::Real) = (x = betainvlogccdf(d.α, d.β, p); x / (1 - x))


#### Sampling

function rand(d::BetaPrime)
    (α, β) = params(d)
    rand(Gamma(α)) / rand(Gamma(β))
end
