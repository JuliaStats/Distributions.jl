immutable Dirichlet <: ContinuousMultivariateDistribution
    alpha::Vector{Float64}
    alpha0::Float64
    lmnB::Float64

    function Dirichlet{T <: Real}(alpha::Vector{T})
        alpha0::Float64 = 0.0
        lmnB::Float64 = 0.0        
        for i in 1:length(alpha)
            ai = alpha[i]
            ai >= 0.0 || throw(DomainError())
            alpha0 += ai
            lmnB += lgamma(ai)
        end
        lmnB -= lgamma(alpha0)
        new(float64(alpha), alpha0, lmnB)
    end

    function Dirichlet(d::Integer, alpha::Float64)
        alpha0 = alpha * d
        new(fill(alpha, d), alpha0, lgamma(alpha) * d - lgamma(alpha0))
    end

    Dirichlet(d::Integer, alpha::Real) = Dirichlet(d, float64(alpha))
end

dim(d::Dirichlet) = length(d.alpha)

mean(d::Dirichlet) = d.alpha .* inv(d.alpha0)

function var(d::Dirichlet)
    α = d.alpha
    α0 = d.alpha0
    c = 1.0 / (α0 * α0 * (α0 + 1.0))

    k = length(α)
    v = Array(Float64, k)
    for i = 1:k
        v[i] = α[i] * (α0 - α[i]) * c
    end
    v
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
            C[i,j] = - α[i] * αjc
        end
        C[j,j] = αj * (α0 - αj) * c
    end

    for j = 1:k-1
        for i = j+1:k
            C[i,j] = C[j,i]
        end
    end
    C
end

function entropy(d::Dirichlet)
    α = d.alpha
    α0 = d.alpha0
    k = length(α)

    en = d.lmnB + (α0 - k) * digamma(α0)
    for j in 1:k
        en -= (α[j] - 1.0) * digamma(α[j])
    end
    return en
end

function modes(d::Dirichlet)
    k = length(d.alpha)
    x = Array(Float64, k)
    s = d.alpha0 - k
    for i in 1:k
        if d.alpha[i] <= 1.0
            error("modes only defined when alpha[i] > 1 for all i")
        end
        x[i] = (d.alpha[i] - 1.0) / s
    end
    return [x]
end

function insupport{T <: Real}(d::Dirichlet, x::Vector{T})
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

pdf{T <: Real}(d::Dirichlet, x::Vector{T}) = exp(logpdf(d, x))

function logpdf{T <: Real}(d::Dirichlet, x::Vector{T})
    a = d.alpha
    k = length(a)
    if length(x) != k
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    s = 0.
    for i in 1 : k
        s += (a[i] - 1.0) * log(x[i])
    end
    s - d.lmnB
end

function logpdf!{T <: Real}(r::AbstractArray, d::Dirichlet, x::Matrix{T})
    a = d.alpha
    if size(x, 1) != length(d.alpha)
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    n = size(x, 2)
    if length(r) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    b::Float64 = d.lmnB
    At_mul_B(r, log(x), d.alpha - 1.0)
    for i in 1:n
        r[i] -= b
    end
    r
end

function rand!(d::Dirichlet, x::Vector)
    s = 0.0
    n = length(x)
    α = d.alpha
    for i in 1:n
        s += (x[i] = randg(α[i]))
    end
    multiply!(x, inv(s)) # this returns x
end

rand(d::Dirichlet) = rand!(d, Array(Float64, dim(d)))

function rand!(d::Dirichlet, X::Matrix)
    k = size(X, 1)
    n = size(X, 2)
    if k != dim(d)
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    α = d.alpha
    for j = 1:n
        s = 0.
        for i = 1:k
            s += (X[i,j] = randg(α[i]))
        end
        inv_s = 1.0 / s
        for i = 1:k
            X[i,j] *= inv_s
        end
    end

    return X
end


#####
#
#  Algorithm: Newton-Raphson
#  
#####

function fit_mle!(
    dty::Type{Dirichlet}, 
    α::Vector{Float64},   # initial guess of α
    Elogp::Vector{Float64};   # expectation/mean of log(p)
    maxiter::Int=25, tol::Float64=1.0e-8, debug::Bool=false)

    K = length(α)
    if length(Elogp) != K
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    g = Array(Float64, K)
    iq = Array(Float64, K)
    α0 = sum(α)

    if debug
        objv = dot(α - 1.0, Elogp) + lgamma(α0) - sum(lgamma(α))
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
            ak = α[k]
            g[k] = gk = digam_α0 - digamma(ak) + Elogp[k]
            iq[k] = - 1.0 / trigamma(ak)

            b += gk * iq[k]
            iqs += iq[k]

            agk = abs(gk)
            if agk > gnorm
                gnorm = agk
            end
        end
        b /= (iz + iqs)

        # update α

        for k = 1:K
            α[k] -= (g[k] - b) * iq[k]
        end
        α0 = sum(α)

        if debug
            prev_objv = objv
            objv = dot(α - 1.0, Elogp) + lgamma(α0) - sum(lgamma(α))
            @printf("Iter %4d: objv = %.4e  ch = %.3e  gnorm = %.3e\n", 
                t, objv, objv - prev_objv, gnorm)
        end

        # determine convergence

        converged = gnorm < tol
    end

    Dirichlet(α)
end


function fit_mle{T <: Real}(::Type{Dirichlet}, P::Matrix{T}; 
    init::Union(Vector{Float64},Nothing)=nothing, 
    maxiter::Int=25, tol::Float64=1.0e-8)

    K = size(P, 1)
    n = size(P, 2)
    c = inv(n)

    # Compute sufficient statistics E[log(p)]

    Elp = Array(Float64, K)
    for i = 1:n
        for k = 1:K
            Elp[k] += log(P[k,i])
        end
    end
    multiply!(Elp, c)

    if init == nothing
        # Initialize mean

        μ = Array(Float64, K)  # E[p]
        γ = Array(Float64, K)  # E[p^2]
    
        for i = 1:n
            for k = 1:K
                pk = P[k, i]
                μ[k] += pk
                γ[k] += pk^2
            end
        end

        for k = 1:K
            μ[k] *= c
            γ[k] *= c
        end

        # Initialize concentration

        α0 = 0.
        for k = 1:K
            μk = μ[k]
            γk = γ[k]
            ak = (μk - γk) / (γk - μk * μk)
            α0 += ak
        end
        α0 /= K

        # store initial solution to μ
        multiply!(μ, α0)
        fit_mle!(Dirichlet, μ, Elp; maxiter=maxiter, tol=tol)        
    else
        fit_mle!(Dirichlet, init, Elp; maxiter=maxiter, tol=tol)
    end
    
end

