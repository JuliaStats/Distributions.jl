"""
    BetaPrime(α, β)

The *Beta prime distribution* has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)}
x^{\\alpha - 1} (1 + x)^{- (\\alpha + \\beta)}, \\quad x > 0
```


The Beta prime distribution is related to the [`Beta`](@ref) distribution via the
relationship that if ``X \\sim \\operatorname{Beta}(\\alpha, \\beta)`` then ``\\frac{X}{1 - X}
\\sim \\operatorname{BetaPrime}(\\alpha, \\beta)``

```julia
BetaPrime()        # equivalent to BetaPrime(1, 1)
BetaPrime(α)       # equivalent to BetaPrime(α, α)
BetaPrime(α, β)    # Beta prime distribution with shape parameters α and β

params(d)          # Get the parameters, i.e. (α, β)
```

External links

* [Beta prime distribution on Wikipedia](http://en.wikipedia.org/wiki/Beta_prime_distribution)

"""
struct BetaPrime{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    BetaPrime{T}(α::T, β::T) where {T} = new{T}(α, β)
end

function BetaPrime(α::T, β::T; check_args::Bool=true) where {T<:Real}
    @check_args BetaPrime (α, α > zero(α)) (β, β > zero(β))
    return BetaPrime{T}(α, β)
end

BetaPrime(α::Real, β::Real; check_args::Bool=true) = BetaPrime(promote(α, β)...; check_args=check_args)
BetaPrime(α::Integer, β::Integer; check_args::Bool=true) = BetaPrime(float(α), float(β); check_args=check_args)
function BetaPrime(α::Real; check_args::Bool=true)
    @check_args BetaPrime (α, α > zero(α))
    BetaPrime(α, α; check_args=false)
end
BetaPrime() = BetaPrime{Float64}(1.0, 1.0)

@distr_support BetaPrime 0.0 Inf

#### Conversions
function convert(::Type{BetaPrime{T}}, α::Real, β::Real) where T<:Real
    BetaPrime(T(α), T(β))
end
Base.convert(::Type{BetaPrime{T}}, d::BetaPrime) where {T<:Real} = BetaPrime{T}(T(d.α), T(d.β))
Base.convert(::Type{BetaPrime{T}}, d::BetaPrime{T}) where {T<:Real} = d

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

function logpdf(d::BetaPrime, x::Real)
    α, β = params(d)
    _x = max(0, x)
    z = xlogy(α - 1, _x) - (α + β) * log1p(_x) - logbeta(α, β)
    return x < 0 ? oftype(z, -Inf) : z
end

function zval(::BetaPrime, x::Real)
    y = max(x, 0)
    z = y / (1 + y)
    # map `Inf` to `Inf` (otherwise it returns `NaN`)
    return isinf(x) && x > 0 ? oftype(z, Inf) : z
end
xval(::BetaPrime, z::Real) = z / (1 - z)

for f in (:cdf, :ccdf, :logcdf, :logccdf)
    @eval $f(d::BetaPrime, x::Real) = $f(Beta(d.α, d.β; check_args=false), zval(d, x))
end

for f in (:quantile, :cquantile, :invlogcdf, :invlogccdf)
    @eval $f(d::BetaPrime, p::Real) = xval(d, $f(Beta(d.α, d.β; check_args=false), p))
end

#### Sampling

function rand(rng::AbstractRNG, d::BetaPrime)
    (α, β) = params(d)
    rand(rng, Gamma(α)) / rand(rng, Gamma(β))
end
