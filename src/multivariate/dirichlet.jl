immutable Dirichlet <: ContinuousMultivariateDistribution
    alpha::Vector{Float64}
    alpha0::Float64

    function Dirichlet{T <: Real}(alpha::Vector{T})
        alpha0::Float64 = 0.0
        for i in 1:length(alpha)
            if alpha[i] < 0.0
                error("Dirichlet: elements of alpha must be non-negative")
            else
                alpha0 += alpha[i]
            end
        end
        new(float64(alpha), alpha0)
    end
end

Dirichlet(d::Integer, alpha::Real) = Dirichlet(fill(alpha, d))

Dirichlet(dim::Integer) = Dirichlet(ones(dim))

dim(d::Dirichlet) = length(d.alpha)

function entropy(d::Dirichlet)
    k = length(d.alpha)
    en = lmnB(d)
    en += (d.alpha0 - k) * digamma(d.alpha0)
    for j in 1:k
        en -= (d.alpha[j] - 1.0) * digamma(d.alpha[j])
    end
    return en
end

function insupport{T <: Real}(d::Dirichlet, x::Vector{T})
    n = length(x)
    if length(d.alpha) != n
        return false
    end
    s = 0.0
    for i in 1:n
        if x[i] < 0.0
            return false
        end
        s += x[i]
    end
    if abs(s - 1.0) > 1e-8
        return false
    end
    return true
end

mean(d::Dirichlet) = d.alpha ./ d.alpha0

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

pdf{T <: Real}(d::Dirichlet, x::Vector{T}) = exp(logpdf(d, x))

function logpdf{T <: Real}(d::Dirichlet, x::Vector{T})
    return dot((d.alpha - 1.0), log(x)) - lmnB(d)
end

function logpdf!{T <: Real}(r::AbstractArray, d::Dirichlet, x::Matrix{T})
    if size(x, 1) != length(d.alpha)
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    n = size(x, 2)
    if length(r) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    b::Float64 = lmnB(d)
    At_mul_B(r, log(x), d.alpha - 1.0)
    for i in 1:n
        r[i] -= b
    end
    r
end

function rand!(d::Dirichlet, x::Vector)
    s = 0.0
    n = length(x)
    for i in 1:n
        tmp = rand(Gamma(d.alpha[i]))
        x[i] = tmp
        s += tmp
    end
    for i in 1:n
        x[i] /= s
    end
    return x
end

function rand(d::Dirichlet)
    x = Array(Float64, length(d.alpha))
    return rand!(d, x)
end

function rand!(d::Dirichlet, X::Matrix)
    m, n = size(X)
    for j in 1:n
        for i in 1:m
            X[i, j] = rand(Gamma(d.alpha[i]))
        end
    end
    for j in 1:n
        isum = 0.0
        for i in 1:m
            isum += X[i, j]
        end
        for i in 1:m
            X[i, j] /= isum
        end
    end
    return X
end

function var(d::Dirichlet)
    n = length(d.alpha)
    tmp = d.alpha0^2 * (d.alpha0 + 1.0)
    S = Array(Float64, n, n)
    for j in 1:n
        for i in 1:n
            if i == j
                S[i, j] = d.alpha[i] * (d.alpha0 - d.alpha[i]) / tmp
            else
                S[i, j] = -d.alpha[i] * d.alpha[j] / tmp
            end
        end
    end
    return S
end

function fit_mle{T <: Real}(::Type{Dirichlet}, P::Matrix{T})
    K, N = size(P)

    alpha = zeros(Float64, K)
    lpbar = zeros(Float64, K)
    E_P = Array(Float64, K)
    E_Psq = zeros(Float64, K)
    g = Array(Float64, K)
    q = Array(Float64, K)

    for i in 1:N
        for k in 1:K
            tmp = P[k, i]
            alpha[k] += tmp
            E_Psq[k] += tmp^2
            lpbar[k] += log(tmp)
        end
    end
    for k in 1:K
        alpha[k] /= N
        E_Psq[k] /= N
        lpbar[k] /= N
    end
    copy!(E_P, alpha)

    alpha0 = (E_P[1] - E_Psq[1]) / (E_Psq[1] - E_P[1]^2)
    for k in 1:K
        alpha[k] *= alpha0
    end

    iteration = 0
    converged = false
    while !converged && iteration < 25
        iteration += 1
        alpha0 = sum(alpha)
        b = 0.0
        iqs = 0.0
        iz = 1.0 / (N * trigamma(alpha0))
        dgalpha0 = digamma(alpha0)
        for k in 1:K
            g[k] = N * (dgalpha0 - digamma(alpha[k]) + lpbar[k])
            q[k] = -N * trigamma(alpha[k])
            b += g[k] / q[k]
            iqs += 1.0 / q[k]
        end
        b /= (iz + iqs)
        for k in 1:K
            alpha[k] -= (g[k] - b) / q[k]
        end
        if norm(g, Inf) > 1e-8
            converged = true
        end
    end

    return Dirichlet(alpha)
end

# Log multinomial beta
function lmnB(d::Dirichlet)
    s = 0.0
    for i in 1:length(d.alpha)
        s += lgamma(d.alpha[i])
    end
    return s - lgamma(d.alpha0)
end

mnB(d::Dirichlet) = exp(lmnB(d))
