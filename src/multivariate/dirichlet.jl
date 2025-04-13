"""
    Dirichlet

The [Dirichlet distribution](http://en.wikipedia.org/wiki/Dirichlet_distribution) is often
used as the conjugate prior for Categorical or Multinomial distributions.
The probability density function of a Dirichlet distribution with parameter
``\\alpha = (\\alpha_1, \\ldots, \\alpha_k)`` is:

```math
f(x; \\alpha) = \\frac{1}{B(\\alpha)} \\prod_{i=1}^k x_i^{\\alpha_i - 1}, \\quad \\text{ with }
B(\\alpha) = \\frac{\\prod_{i=1}^k \\Gamma(\\alpha_i)}{\\Gamma \\left( \\sum_{i=1}^k \\alpha_i \\right)},
\\quad x_1 + \\cdots + x_k = 1
```

```julia
# Let alpha be a vector
Dirichlet(alpha)         # Dirichlet distribution with parameter vector alpha

# Let a be a positive scalar
Dirichlet(k, a)          # Dirichlet distribution with parameter a * ones(k)
```
"""
struct Dirichlet{T<:Real,Ts<:AbstractVector{T},S<:Real} <: ContinuousMultivariateDistribution
    alpha::Ts
    alpha0::T
    lmnB::S

    function Dirichlet{T}(alpha::AbstractVector{T}; check_args::Bool=true) where T
        @check_args(
            Dirichlet,
            (alpha, all(x -> x > zero(x), alpha), "alpha must be a positive vector."),
        )
        alpha0 = sum(alpha)
        lmnB = sum(loggamma, alpha) - loggamma(alpha0)
        new{T,typeof(alpha),typeof(lmnB)}(alpha, alpha0, lmnB)
    end
end

function Dirichlet(alpha::AbstractVector{T}; check_args::Bool=true) where {T<:Real}
    Dirichlet{T}(alpha; check_args=check_args)
end
function Dirichlet(d::Integer, alpha::Real; check_args::Bool=true)
    @check_args Dirichlet (d, d > zero(d)) (alpha, alpha > zero(alpha))
    return Dirichlet{typeof(alpha)}(Fill(alpha, d); check_args=false)
end

struct DirichletCanon{T<:Real,Ts<:AbstractVector{T}}
    alpha::Ts
end

length(d::DirichletCanon) = length(d.alpha)

Base.eltype(::Type{<:Dirichlet{T}}) where {T} = T

#### Conversions
convert(::Type{Dirichlet{T}}, cf::DirichletCanon) where {T<:Real} =
    Dirichlet(convert(AbstractVector{T}, cf.alpha))
convert(::Type{Dirichlet{T}}, alpha::AbstractVector{<:Real}) where {T<:Real} =
    Dirichlet(convert(AbstractVector{T}, alpha))
convert(::Type{Dirichlet{T}}, d::Dirichlet{<:Real}) where {T<:Real} =
    Dirichlet(convert(AbstractVector{T}, d.alpha))

convert(::Type{Dirichlet{T}}, cf::DirichletCanon{T}) where {T<:Real} = Dirichlet(cf.alpha)
convert(::Type{Dirichlet{T}}, alpha::AbstractVector{T}) where {T<:Real} =
    Dirichlet(alpha)
convert(::Type{Dirichlet{T}}, d::Dirichlet{T}) where {T<:Real} = d

Base.show(io::IO, d::Dirichlet) = show(io, d, (:alpha,))

# Properties

length(d::Dirichlet) = length(d.alpha)
mean(d::Dirichlet) = d.alpha .* inv(d.alpha0)
params(d::Dirichlet) = (d.alpha,)
@inline partype(::Dirichlet{T}) where {T<:Real} = T

function var(d::Dirichlet)
    α0 = d.alpha0
    c = inv(α0^2 * (α0 + 1))
    v = map(d.alpha) do αi
        αi * (α0 - αi) * c
    end
    return v
end

function cov(d::Dirichlet)
    α = d.alpha
    α0 = d.alpha0
    c = inv(α0^2 * (α0 + 1))

    T = typeof(zero(eltype(α))^2 * c)
    k = length(α)
    C = Matrix{T}(undef, k, k)
    for j = 1:k
        αj = α[j]
        αjc = αj * c
        for i in 1:(j-1)
            @inbounds C[i,j] = C[j,i]
        end
        @inbounds C[j,j] = (α0 - αj) * αjc
        for i in (j+1):k
            @inbounds C[i,j] = - α[i] * αjc
        end
    end

    return C
end

function entropy(d::Dirichlet)
    α0 = d.alpha0
    α = d.alpha
    k = length(d.alpha)
    en = d.lmnB + (α0 - k) * digamma(α0) - sum(αj -> (αj - 1) * digamma(αj), α)
    return en
end

function dirichlet_mode!(r::AbstractVector{<:Real}, α::AbstractVector{<:Real}, α0::Real)
    all(x -> x > 1, α) || error("Dirichlet has a mode only when alpha[i] > 1 for all i")
    k = length(α)
    inv_s = inv(α0 - k)
    @. r = inv_s * (α - 1)
    return r
end

function dirichlet_mode(α::AbstractVector{<:Real}, α0::Real)
    all(x -> x > 1, α) || error("Dirichlet has a mode only when alpha[i] > 1 for all i")
    inv_s = inv(α0 - length(α))
    r = map(α) do αi
        inv_s * (αi - 1)
    end
    return r
end

mode(d::Dirichlet) = dirichlet_mode(d.alpha, d.alpha0)
mode(d::DirichletCanon) = dirichlet_mode(d.alpha, sum(d.alpha))

modes(d::Dirichlet) = [mode(d)]


# Evaluation

function insupport(d::Dirichlet, x::AbstractVector{<:Real})
    return length(d) == length(x) && !any(x -> x < zero(x), x) && sum(x) ≈ 1
end

function _logpdf(d::Dirichlet, x::AbstractVector{<:Real})
    if !insupport(d, x)
        return xlogy(one(eltype(d.alpha)), zero(eltype(x))) - d.lmnB
    end
    a = d.alpha
    s = sum(xlogy(αi - 1, xi) for (αi, xi) in zip(d.alpha, x))
    return s - d.lmnB
end

# sampling

function _rand!(rng::AbstractRNG,
                d::Union{Dirichlet,DirichletCanon},
                x::AbstractVector{<:Real})
    for (i, αi) in zip(eachindex(x), d.alpha)
        @inbounds x[i] = rand(rng, Gamma(αi))
    end
    lmul!(inv(sum(x)), x) # this returns x
end

function _rand!(rng::AbstractRNG,
                d::Dirichlet{T,<:FillArrays.AbstractFill{T}},
                x::AbstractVector{<:Real}) where {T<:Real}
    rand!(rng, Gamma(FillArrays.getindex_value(d.alpha)), x)
    lmul!(inv(sum(x)), x) # this returns x
end

#######################################
#
#  Estimation
#
#######################################

struct DirichletStats <: SufficientStats
    slogp::Vector{Float64}   # (weighted) sum of log(p)
    tw::Float64              # total sample weights

    DirichletStats(slogp::Vector{Float64}, tw::Real) = new(slogp, Float64(tw))
end

length(ss::DirichletStats) = length(s.slogp)

mean_logp(ss::DirichletStats) = ss.slogp * inv(ss.tw)

function suffstats(::Type{<:Dirichlet}, P::AbstractMatrix{Float64})
    K = size(P, 1)
    n = size(P, 2)
    slogp = zeros(K)
    for i = 1:n
        for k = 1:K
            @inbounds slogp[k] += log(P[k,i])
        end
    end
    DirichletStats(slogp, n)
end

function suffstats(::Type{<:Dirichlet}, P::AbstractMatrix{Float64},
                   w::AbstractArray{Float64})
    K = size(P, 1)
    n = size(P, 2)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end

    tw = 0.
    slogp = zeros(K)

    for i = 1:n
        @inbounds wi = w[i]
        tw += wi
        for k = 1:K
            @inbounds slogp[k] += log(P[k,i]) * wi
        end
    end
    DirichletStats(slogp, tw)
end

# fit_mle methods

## Initialization

function _dirichlet_mle_init2(μ::Vector{Float64}, γ::Vector{Float64})
    K = length(μ)

    α0 = 0.
    for k = 1:K
        @inbounds μk = μ[k]
        @inbounds γk = γ[k]
        ak = (μk - γk) / (γk - μk * μk)
        α0 += ak
    end
    α0 /= K

    lmul!(α0, μ)
end

function dirichlet_mle_init(P::AbstractMatrix{Float64})
    K = size(P, 1)
    n = size(P, 2)

    μ = vec(sum(P, dims=2))       # E[p]
    γ = vec(sum(abs2, P, dims=2)) # E[p^2]

    c = 1.0 / n
    μ .*= c
    γ .*= c

    _dirichlet_mle_init2(μ, γ)
end

function dirichlet_mle_init(P::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    K = size(P, 1)
    n = size(P, 2)

    μ = zeros(K)  # E[p]
    γ = zeros(K)  # E[p^2]
    tw = 0.0

    for i in 1:n
        @inbounds wi = w[i]
        tw += wi
        for k in 1:K
            pk = P[k, i]
            @inbounds μ[k] += pk * wi
            @inbounds γ[k] += pk * pk * wi
        end
    end

    c = 1.0 / tw
    μ .*= c
    γ .*= c

    _dirichlet_mle_init2(μ, γ)
end

## Newton-Ralphson algorithm

function fit_dirichlet!(elogp::Vector{Float64}, α::Vector{Float64};
    maxiter::Int=25, tol::Float64=1.0e-12, debug::Bool=false)
    # This function directly overrides α

    K = length(elogp)
    length(α) == K || throw(DimensionMismatch("Inconsistent argument dimensions."))

    g = Vector{Float64}(undef, K)
    iq = Vector{Float64}(undef, K)
    α0 = sum(α)

    if debug
        objv = dot(α .- 1.0, elogp) + loggamma(α0) - sum(loggamma, α)
    end

    t = 0
    converged = false
    while !converged && t < maxiter
        t += 1

        # compute gradient & Hessian
        # (b is computed as well)

        digam_α0 = digamma(α0)
        iz = 1.0 / trigamma(α0)
        gnorm = 0.
        b = 0.
        iqs = 0.

        for k = 1:K
            @inbounds ak = α[k]
            @inbounds g[k] = gk = digam_α0 - digamma(ak) + elogp[k]
            @inbounds iq[k] = - 1.0 / trigamma(ak)

            @inbounds b += gk * iq[k]
            @inbounds iqs += iq[k]

            agk = abs(gk)
            if agk > gnorm
                gnorm = agk
            end
        end
        b /= (iz + iqs)

        # update α

        for k = 1:K
            @inbounds α[k] -= (g[k] - b) * iq[k]
            @inbounds if α[k] < 1.0e-12
                α[k] = 1.0e-12
            end
        end
        α0 = sum(α)

        if debug
            prev_objv = objv
            objv = dot(α .- 1.0, elogp) + loggamma(α0) - sum(loggamma, α)
            @printf("Iter %4d: objv = %.4e  ch = %.3e  gnorm = %.3e\n",
                t, objv, objv - prev_objv, gnorm)
        end

        # determine convergence

        converged = gnorm < tol
    end

    if !converged
        throw(ErrorException("No convergence after $maxiter (maxiter) iterations."))
    end

    Dirichlet(α)
end


function fit_mle(::Type{T}, P::AbstractMatrix{Float64};
    init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12,
    debug::Bool=false) where {T<:Dirichlet}

    α = isempty(init) ? dirichlet_mle_init(P) : init
    elogp = mean_logp(suffstats(T, P))
    fit_dirichlet!(elogp, α; maxiter=maxiter, tol=tol, debug=debug)
end

function fit_mle(::Type{<:Dirichlet}, P::AbstractMatrix{Float64},
                 w::AbstractArray{Float64};
    init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12,
    debug::Bool=false)

    n = size(P, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions."))

    α = isempty(init) ? dirichlet_mle_init(P, w) : init
    elogp = mean_logp(suffstats(Dirichlet, P, w))
    fit_dirichlet!(elogp, α; maxiter=maxiter, tol=tol, debug=debug)
end
