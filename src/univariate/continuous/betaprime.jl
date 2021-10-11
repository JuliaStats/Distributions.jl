"""
    BetaPrime <: ContinuousUnivariateDistribution

The *Beta prime distribution* has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)}
x^{\\alpha - 1} (1 + x)^{- (\\alpha + \\beta)}, \\quad x > 0
```

The Beta prime distribution is related to the [`Beta`](@ref) distribution via the
relation ship that if ``X \\sim \\operatorname{Beta}(\\alpha, \\beta)`` then ``\\frac{X}{1 - X}
\\sim \\operatorname{BetaPrime}(\\alpha, \\beta)``

```julia
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

# constructors with positional arguments
function BetaPrime(α::T, β::T; check_args::Bool=true) where {T<:Real}
    check_args && @check_args(BetaPrime, α > zero(α) && β > zero(β))
    return BetaPrime{T}(α, β)
end
BetaPrime(α::Real, β::Real=α; kwargs...) = BetaPrime(promote(α, β)...; kwargs...)
BetaPrime(α::Integer, β::Integer; kwargs...) = BetaPrime(float(α), float(β); kwargs...)

# constructor with keyword arguments and ASCII alternatives
"""
    BetaPrime(; α::Real=1.0, β::Real=α, check_args::Bool=true)

Construct a [`BetaPrime`](@ref) distribution with parameters `α` and `β`.

Use `check_args=false` to bypass the check if `α` and `β` are positive.

# ASCII keyword arguments

You can use `alpha` and `beta` instead of `α` and `β` to specify the parameters
of the Beta prime distribution. The Unicode names have higher precedence, i.e., if
both `α` and `alpha` or `β` and `beta` are given a Beta prime distribution with
parameters `α` and `β` is constructed.
"""
function BetaPrime(; alpha::Real=1.0, α::Real=alpha, beta::Real=α, β::Real=beta, kwargs...)
    return BetaPrime(α, β; kwargs...)
end

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
