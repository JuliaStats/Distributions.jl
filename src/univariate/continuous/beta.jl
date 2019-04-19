"""
    Beta <: ContinuousUnivariateDistribution

The *beta* probability distribution.

# Constructors

    Beta(α|alpha=1, β|beta=1)

Construct a `Beta` distribution object with parameters `α` and `β`.

    Beta(mean=,var=)
    Beta(mean=,std=)

Construct a `Beta` distribution object matching the relevant moments.

# Details

The beta distribution has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)}
 x^{\\alpha - 1} (1 - x)^{\\beta - 1}, \\quad x \\in [0, 1]
```

The Beta distribution is related to the [`Gamma`](@ref) distribution via the
property that if ``X \\sim \\operatorname{Gamma}(\\alpha)`` and ``Y \\sim \\operatorname{Gamma}(\\beta)``
independently, then ``X / (X + Y) \\sim Beta(\\alpha, \\beta)``.

# Examples

```julia
Beta()
Beta(α=3, β=4)
Beta(mean=0.2, std=0.1)
```

# External links

* [Beta distribution on Wikipedia](http://en.wikipedia.org/wiki/Beta_distribution)

"""
struct Beta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T

    function Beta{T}(α::T, β::T) where T
        @check_args(Beta, α > zero(α) && β > zero(β))
        new{T}(α, β)
    end
end

Beta(α::T, β::T) where {T<:Real} = Beta{T}(α, β)
Beta(α::Real, β::Real) = Beta(promote(α, β)...)
Beta(α::Integer, β::Integer) = Beta(float(α), float(β))

@kwdispatch (::Type{D})(;alpha=>α, beta=>β) where {D<:Beta} begin
    () -> D(1,1)
    (β) -> D(1,β)
    (α) -> D(α,1)
    (α,β) -> D(α,β)

    function (mean, var)
        @check_args(Beta, 0 < mean < 1)
        @check_args(Beta, 0 < var < mean*(1-mean))
        U = (mean*(1-mean))/var - 1
        α = mean*U
        β = U-α
        D(α,β)
    end

    function (mean, std)
        D(mean=mean,var=sqrt(std))
    end
end

@distr_support Beta 0.0 1.0

#### Conversions
function convert(::Type{Beta{T}}, α::Real, β::Real) where T<:Real
    Beta(T(α), T(β))
end
function convert(::Type{Beta{T}}, d::Beta{S}) where {T <: Real, S <: Real}
    Beta(T(d.α), T(d.β))
end

#### Parameters

params(d::Beta) = (d.α, d.β)
@inline partype(d::Beta{T}) where {T<:Real} = T


#### Statistics

mean(d::Beta) = ((α, β) = params(d); α / (α + β))

function mode(d::Beta)
    (α, β) = params(d)
    (α > 1 && β > 1) || error("mode is defined only when α > 1 and β > 1.")
    return (α - 1) / (α + β - 2)
end

modes(d::Beta) = [mode(d)]

function var(d::Beta)
    (α, β) = params(d)
    s = α + β
    return (α * β) / (abs2(s) * (s + 1))
end

meanlogx(d::Beta) = ((α, β) = params(d); digamma(α) - digamma(α + β))

varlogx(d::Beta) = ((α, β) = params(d); trigamma(α) - trigamma(α + β))
stdlogx(d::Beta) = sqrt(varlogx(d))

function skewness(d::Beta)
    (α, β) = params(d)
    if α == β
        return zero(α)
    else
        s = α + β
        (2(β - α) * sqrt(s + 1)) / ((s + 2) * sqrt(α * β))
    end
end

function kurtosis(d::Beta)
    α, β = params(d)
    s = α + β
    p = α * β
    6(abs2(α - β) * (s + 1) - p * (s + 2)) / (p * (s + 2) * (s + 3))
end

function entropy(d::Beta)
    α, β = params(d)
    s = α + β
    lbeta(α, β) - (α - 1) * digamma(α) - (β - 1) * digamma(β) +
        (s - 2) * digamma(s)
end


#### Evaluation

@_delegate_statsfuns Beta beta α β

gradlogpdf(d::Beta{T}, x::Real) where {T<:Real} =
    ((α, β) = params(d); 0 <= x <= 1 ? (α - 1) / x - (β - 1) / (1 - x) : zero(T))


#### Sampling

struct BetaSampler{T<:Real, S1 <: Sampleable{Univariate,Continuous},
                   S2 <: Sampleable{Univariate,Continuous}} <:
    Sampleable{Univariate,Continuous}
    γ::Bool
    iα::T
    iβ::T
    s1::S1
    s2::S2
end

function sampler(d::Beta{T}) where T
    (α, β) = params(d)
    if (α ≤ 1.0) && (β ≤ 1.0)
        return BetaSampler(false, inv(α), inv(β),
                           sampler(Uniform()), sampler(Uniform()))
    else
        return BetaSampler(true, inv(α), inv(β),
                           sampler(Gamma(α, one(T))),
                           sampler(Gamma(β, one(T))))
    end
end

# From Knuth
function rand(rng::AbstractRNG, s::BetaSampler)
    if s.γ
        g1 = rand(rng, s.s1)
        g2 = rand(rng, s.s2)
        return g1 / (g1 + g2)
    else
        iα = s.iα
        iβ = s.iβ
        while true
            u = rand(rng) # the Uniform sampler just calls rand()
            v = rand(rng)
            x = u^iα
            y = v^iβ
            if x + y ≤ one(x)
                if (x + y > 0)
                    return x / (x + y)
                else
                    logX = log(u) * iα
                    logY = log(v) * iβ
                    logM = logX > logY ? logX : logY
                    logX -= logM
                    logY -= logM
                    return exp(logX - log(exp(logX) + exp(logY)))
                end
            end
        end
    end
end

function rand(rng::AbstractRNG, d::Beta{T}) where T
    (α, β) = params(d)
    if (α ≤ 1.0) && (β ≤ 1.0)
        while true
            u = rand(rng)
            v = rand(rng)
            x = u^inv(α)
            y = v^inv(β)
            if x + y ≤ one(x)
                if (x + y > 0)
                    return x / (x + y)
                else
                    logX = log(u) / α
                    logY = log(v) / β
                    logM = logX > logY ? logX : logY
                    logX -= logM
                    logY -= logM
                    return exp(logX - log(exp(logX) + exp(logY)))
                end
            end
        end
    else
        g1 = rand(rng, Gamma(α, one(T)))
        g2 = rand(rng, Gamma(β, one(T)))
        return g1 / (g1 + g2)
    end
end

#### Fit model

# TODO: add MLE method (should be similar to Dirichlet)

# This is a moment-matching method (not MLE)
#
function fit(::Type{Beta}, x::AbstractArray{T}) where T<:Real
    x_bar = mean(x)
    v_bar = varm(x, x_bar)
    α = x_bar * (((x_bar * (1 - x_bar)) / v_bar) - 1)
    β = (1 - x_bar) * (((x_bar * (1 - x_bar)) / v_bar) - 1)
    Beta(α, β)
end
