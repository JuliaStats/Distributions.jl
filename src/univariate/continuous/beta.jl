"""
    Beta(α, β)

The *Beta distribution* has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)}
 x^{\\alpha - 1} (1 - x)^{\\beta - 1}, \\quad x \\in [0, 1]
```

The Beta distribution is related to the [`Gamma`](@ref) distribution via the
property that if ``X \\sim \\operatorname{Gamma}(\\alpha)`` and ``Y \\sim \\operatorname{Gamma}(\\beta)``
independently, then ``X / (X + Y) \\sim \\operatorname{Beta}(\\alpha, \\beta)``.


```julia
Beta()        # equivalent to Beta(1, 1)
Beta(α)       # equivalent to Beta(α, α)
Beta(α, β)    # Beta distribution with shape parameters α and β

params(d)     # Get the parameters, i.e. (α, β)
```

External links

* [Beta distribution on Wikipedia](http://en.wikipedia.org/wiki/Beta_distribution)

"""
struct Beta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    Beta{T}(α::T, β::T) where {T} = new{T}(α, β)
end

function Beta(α::T, β::T; check_args::Bool=true) where {T<:Real}
    @check_args Beta (α, α > zero(α)) (β, β > zero(β))
    return Beta{T}(α, β)
end

Beta(α::Real, β::Real; check_args::Bool=true) = Beta(promote(α, β)...; check_args=check_args)
Beta(α::Integer, β::Integer; check_args::Bool=true) = Beta(float(α), float(β); check_args=check_args)
function Beta(α::Real; check_args::Bool=true)
    @check_args Beta (α, α > zero(α))
    Beta(α, α; check_args=false)
end
Beta() = Beta{Float64}(1.0, 1.0)

@distr_support Beta 0.0 1.0

#### Conversions
function convert(::Type{Beta{T}}, α::Real, β::Real) where T<:Real
    Beta(T(α), T(β))
end
Base.convert(::Type{Beta{T}}, d::Beta) where {T<:Real} = Beta{T}(T(d.α), T(d.β))
Base.convert(::Type{Beta{T}}, d::Beta{T}) where {T<:Real} = d

#### Parameters

params(d::Beta) = (d.α, d.β)
@inline partype(d::Beta{T}) where {T<:Real} = T


#### Statistics

mean(d::Beta) = ((α, β) = params(d); α / (α + β))

function mode(d::Beta; check_args::Bool=true)
    α, β = params(d)
    @check_args(
        Beta,
        (α, α > 1, "mode is defined only when α > 1."),
        (β, β > 1, "mode is defined only when β > 1."),
    )
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

function kldivergence(p::Beta, q::Beta)
    αp, βp = params(p)
    αq, βq = params(q)
    return logbeta(αq, βq) - logbeta(αp, βp) + (αp - αq) * digamma(αp) +
        (βp - βq) * digamma(βp) + (αq - αp + βq - βp) * digamma(αp + βp)
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
"""
    fit_mle(::Type{<:Beta}, x::AbstractArray{T})

Maximum Likelihood Estimate of `Beta` Distribution via Newton's Method
"""
function fit_mle(::Type{<:Beta}, x::AbstractArray{T};
    maxiter::Int=1000, tol::Float64=1e-14) where T<:Real

    α₀,β₀ = params(fit(Beta,x)) #initial guess of parameters
    g₁ = mean(log.(x))
    g₂ = mean(log.(one(T) .- x))
    θ= [α₀ ; β₀ ]

    converged = false
    t=0
    while !converged && t < maxiter #newton method
        t+=1
        temp1 = digamma(θ[1]+θ[2])
        temp2 = trigamma(θ[1]+θ[2])
        grad = [g₁+temp1-digamma(θ[1])
               temp1+g₂-digamma(θ[2])]
        hess = [temp2-trigamma(θ[1]) temp2
                temp2 temp2-trigamma(θ[2])]
        Δθ = hess\grad #newton step
        θ .-= Δθ
        converged = dot(Δθ,Δθ) < 2*tol #stopping criterion
    end

    return Beta(θ[1], θ[2])
end

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
