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
    return
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

# Log multinomial beta
function lmnB(d::Dirichlet)
    s = 0.0
    for i in 1:length(d.alpha)
        s += lgamma(d.alpha[i])
    end
    return s - lgamma(d.alpha0)
end

mnB(d::Dirichlet) = exp(lmnB(d))
