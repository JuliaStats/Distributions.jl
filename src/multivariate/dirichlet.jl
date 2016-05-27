immutable Dirichlet{T <: Real} <: ContinuousMultivariateDistribution
    alpha::Vector{T}
    alpha0::T
    lmnB::T

    function Dirichlet(alpha::Vector{T})
        alpha0::T = zero(T)
        lmnB::T = zero(T)
        for i in 1:length(alpha)
            ai = alpha[i]
            ai > 0 || throw(ArgumentError("Dirichlet: alpha must be a positive vector."))
            alpha0 += ai
            lmnB += lgamma(ai)
        end
        lmnB -= lgamma(alpha0)
        new(alpha, alpha0, lmnB)
    end

    function Dirichlet(d::Integer, alpha::T)
        alpha0 = alpha * d
        new(fill(alpha, d), alpha0, lgamma(alpha) * d - lgamma(alpha0))
    end
end

Dirichlet{T <: Real}(alpha::Vector{T}) = Dirichlet{T}(alpha)
Dirichlet{T <: Real}(d::Integer, alpha::T) = Dirichlet{T}(d, alpha)
Dirichlet{T <: Integer}(alpha::Vector{T}) = Dirichlet{Float64}(alpha)
Dirichlet(d::Integer, alpha::Integer) = Dirichlet{Float64}(d, Float64(alpha))

immutable DirichletCanon
    alpha::Vector{Float64}
end

length(d::DirichletCanon) = length(d.alpha)

#### Conversions
convert(::Type{Dirichlet{Float64}}, cf::DirichletCanon) = Dirichlet(cf.alpha)
convert{T <: Real, S <: Real}(::Type{Dirichlet{T}}, alpha::Vector{S}) = Dirichlet(convert(Vector{T}, alpha))
convert{T <: Real, S <: Real}(::Type{Dirichlet{T}}, d::Dirichlet{S}) = Dirichlet(convert(Vector{T}, d.alpha))



Base.show(io::IO, d::Dirichlet) = show(io, d, (:alpha,))

# Properties

length(d::Dirichlet) = length(d.alpha)
mean(d::Dirichlet) = d.alpha .* inv(d.alpha0)
params(d::Dirichlet) = (d.alpha,)

function var(d::Dirichlet)
    α = d.alpha
    α0 = d.alpha0
    c = 1.0 / (α0 * α0 * (α0 + 1.0))

    k = length(α)
    v = Array(Float64, k)
    for i = 1:k
        @inbounds αi = α[i]
        @inbounds v[i] = αi * (α0 - αi) * c
    end
    return v
end

function cov(d::Dirichlet)
    α = d.alpha
    α0 = d.alpha0
    c = 1.0 / (α0 * α0 * (α0 + 1.0))

    k = length(α)
    C = Array(Float64, k, k)

    for j = 1:k
        αj = α[j]
        αjc = αj * c
        for i = 1:j-1
            @inbounds C[i,j] = - α[i] * αjc
        end
        @inbounds C[j,j] = αj * (α0 - αj) * c
    end

    for j = 1:k-1, i = j+1:k
        @inbounds C[i,j] = C[j,i]
    end
    return C
end

function entropy(d::Dirichlet)
    α = d.alpha
    α0 = d.alpha0
    k = length(α)

    en = d.lmnB + (α0 - k) * digamma(α0)
    for j in 1:k
        @inbounds αj = α[j]
        en -= (αj - 1.0) * digamma(αj)
    end
    return en
end


function dirichlet_mode!{T <: Real}(r::Vector{T}, α::Vector{T}, α0::T)
    k = length(α)
    s = α0 - k
    for i = 1:k
        @inbounds αi = α[i]
        if αi <= one(T)
            error("Dirichlet has a mode only when alpha[i] > 1 for all i" )
        end
        @inbounds r[i] = (αi - one(T)) / s
    end
    return r
end

dirichlet_mode{T <: Real}(α::Vector{T}, α0::T) = dirichlet_mode!(Array(T, length(α)), α, α0)

mode(d::Dirichlet) = dirichlet_mode(d.alpha, d.alpha0)
mode(d::DirichletCanon) = dirichlet_mode(d.alpha, sum(d.alpha))

modes(d::Dirichlet) = [mode(d)]


# Evaluation

function insupport{T<:Real}(d::Dirichlet, x::AbstractVector{T})
    n = length(x)
    if length(d.alpha) != n
        return false
    end
    s = 0.0
    for i in 1:n
        xi = x[i]
        if xi < 0.0
            return false
        end
        s += xi
    end
    if abs(s - 1.0) > 1e-8
        return false
    end
    return true
end

function _logpdf{T<:Real}(d::Dirichlet, x::AbstractVector{T})
    a = d.alpha
    s = 0.
    for i in 1:length(a)
        @inbounds s += (a[i] - 1.0) * log(x[i])
    end
    return s - d.lmnB
end

# sampling

function _rand!{T<:Real}(d::Union{Dirichlet,DirichletCanon}, x::AbstractVector{T})
    s = 0.0
    n = length(x)
    α = d.alpha
    for i in 1:n
        @inbounds s += (x[i] = rand(Gamma(α[i])))
    end
    multiply!(x, inv(s)) # this returns x
end

#######################################
#
#  Estimation
#
#######################################

immutable DirichletStats <: SufficientStats
    slogp::Vector{Float64}   # (weighted) sum of log(p)
    tw::Float64              # total sample weights

    DirichletStats(slogp::Vector{Float64}, tw::Real) = new(slogp, Float64(tw))
end

length(ss::DirichletStats) = length(s.slogp)

mean_logp(ss::DirichletStats) = ss.slogp * inv(ss.tw)

function suffstats(::Type{Dirichlet}, P::AbstractMatrix{Float64})
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

function suffstats(::Type{Dirichlet}, P::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    K = size(P, 1)
    n = size(P, 2)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
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

    multiply!(μ, α0)
end

function dirichlet_mle_init(P::AbstractMatrix{Float64})
    K = size(P, 1)
    n = size(P, 2)

    μ = Array(Float64, K)  # E[p]
    γ = Array(Float64, K)  # E[p^2]

    for i = 1:n
        for k = 1:K
            @inbounds pk = P[k, i]
            @inbounds μ[k] += pk
            @inbounds γ[k] += pk * pk
        end
    end

    c = 1.0 / n
    for k = 1:K
        μ[k] *= c
        γ[k] *= c
    end

    _dirichlet_mle_init2(μ, γ)
end

function dirichlet_mle_init(P::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    K = size(P, 1)
    n = size(P, 2)

    μ = Array(Float64, K)  # E[p]
    γ = Array(Float64, K)  # E[p^2]
    tw = 0.

    for i = 1:n
        @inbounds wi = w[i]
        tw += wi
        for k = 1:K
            pk = P[k, i]
            @inbounds μ[k] += pk * wi
            @inbounds γ[k] += pk * pk * wi
        end
    end

    c = 1.0 / tw
    for k = 1:K
        μ[k] *= c
        γ[k] *= c
    end

    _dirichlet_mle_init2(μ, γ)
end

## Newton-Ralphson algorithm

function fit_dirichlet!(elogp::Vector{Float64}, α::Vector{Float64};
    maxiter::Int=25, tol::Float64=1.0e-12, debug::Bool=false)
    # This function directly overrides α

    K = length(elogp)
    length(α) == K || throw(ArgumentError("Inconsistent argument dimensions."))

    g = Array(Float64, K)
    iq = Array(Float64, K)
    α0 = sum(α)

    if debug
        objv = dot(α - 1.0, elogp) + lgamma(α0) - sum(lgamma(α))
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
            objv = dot(α - 1.0, elogp) + lgamma(α0) - sum(lgamma(α))
            @printf("Iter %4d: objv = %.4e  ch = %.3e  gnorm = %.3e\n",
                t, objv, objv - prev_objv, gnorm)
        end

        # determine convergence

        converged = gnorm < tol
    end

    Dirichlet(α)
end


function fit_mle(::Type{Dirichlet}, P::AbstractMatrix{Float64};
    init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12)

    α = isempty(init) ? dirichlet_mle_init(P) : init
    elogp = mean_logp(suffstats(Dirichlet, P))
    fit_dirichlet!(elogp, α; maxiter=maxiter, tol=tol)
end

function fit_mle(::Type{Dirichlet}, P::AbstractMatrix{Float64}, w::AbstractArray{Float64};
    init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12)

    n = size(P, 2)
    length(w) == n || throw(ArgumentError("Inconsistent argument dimensions."))

    α = isempty(init) ? dirichlet_mle_init(P, w) : init
    elogp = mean_logp(suffstats(Dirichlet, P, w))
    fit_dirichlet!(elogp, α; maxiter=maxiter, tol=tol)
end
