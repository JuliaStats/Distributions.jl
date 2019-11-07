"""
    Beta(α,β)

The *Beta distribution* has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)}
 x^{\\alpha - 1} (1 - x)^{\\beta - 1}, \\quad x \\in [0, 1]
```

The Beta distribution is related to the [`Gamma`](@ref) distribution via the
property that if ``X \\sim \\operatorname{Gamma}(\\alpha)`` and ``Y \\sim \\operatorname{Gamma}(\\beta)``
independently, then ``X / (X + Y) \\sim Beta(\\alpha, \\beta)``.


```julia
Beta()        # equivalent to Beta(1, 1)
Beta(a)       # equivalent to Beta(a, a)
Beta(a, b)    # Beta distribution with shape parameters a and b

params(d)     # Get the parameters, i.e. (a, b)
```

External links

* [Beta distribution on Wikipedia](http://en.wikipedia.org/wiki/Beta_distribution)

"""
struct Beta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    Beta{T}(α::T, β::T) where {T} = new{T}(α, β)
end

function Beta(α::T, β::T; check_args=true) where {T<:Real}
    check_args && @check_args(Beta, α > zero(α) && β > zero(β))
    return Beta{T}(α, β)
end

Beta(α::Real, β::Real) = Beta(promote(α, β)...)
Beta(α::Integer, β::Integer) = Beta(float(α), float(β))
Beta(α::Real) = Beta(α, α)
Beta() = Beta(1.0, 1.0, check_args=false)

@distr_support Beta 0.0 1.0

#### Conversions
function convert(::Type{Beta{T}}, α::Real, β::Real) where T<:Real
    Beta(T(α), T(β))
end
function convert(::Type{Beta{T}}, d::Beta{S}) where {T <: Real, S <: Real}
    Beta(T(d.α), T(d.β), check_args=false)
end

#### Parameters

params(d::Beta) = (d.α, d.β)
@inline partype(d::Beta{T}) where {T<:Real} = T


#### Statistics

mean(d::Beta) = ((α, β) = params(d); α / (α + β))

function mode(d::Beta; check_args=true)
    (α, β) = params(d)
    if check_args
        (α > 1 && β > 1) || error("mode is defined only when α > 1 and β > 1.")
    end
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
    logbeta(α, β) - (α - 1) * digamma(α) - (β - 1) * digamma(β) +
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

"""
    fit(::Type{<:Beta}, x::AbstractArray{T})

fit a `Beta` distribution
"""
function fit(::Type{<:Beta}, x::AbstractArray{T}) where T<:Real
    x_bar = mean(x)
    v_bar = varm(x, x_bar)
    temp = ((x_bar * (one(T) - x_bar)) / v_bar) - one(T)
    α = x_bar * temp
    β = (one(T) - x_bar) * temp
    return Beta(α, β)
end
