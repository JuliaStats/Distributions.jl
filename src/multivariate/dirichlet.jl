immutable Dirichlet <: ContinuousMultivariateDistribution
    alpha::Vector{Float64}

    function Dirichlet{T <: Real}(alpha::Vector{T})
        for i in 1:length(alpha)
            if alpha[i] < 0.0
                error("Dirichlet: elements of alpha must be non-negative")
            end
        end
        new(float64(alpha))
    end
end

Dirichlet(d::Integer, alpha::Real) = Dirichlet(fill(alpha, d))

Dirichlet(dim::Integer) = Dirichlet(ones(dim))

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

mean(d::Dirichlet) = d.alpha ./ sum(d.alpha)

pdf{T <: Real}(d::Dirichlet, x::Vector{T}) = exp(logpdf(d, x))

function logpdf{T <: Real}(d::Dirichlet, x::Vector{T})
    b = sum(lgamma(d.alpha)) - lgamma(sum(d.alpha))
    return dot((d.alpha - 1), log(x)) - b
end

function logpdf!{T <: Real}(r::AbstractArray, d::Dirichlet, x::Matrix{T})
    if size(x, 1) != length(d.alpha)
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    n = size(x, 2)
    if length(r) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    b::Float64 = sum(lgamma(d.alpha)) - lgamma(sum(d.alpha))  
    At_mul_B(r, log(x), d.alpha - 1.0)
    for i in 1:n
        r[i] -= b
    end
    return
end

function logpdf{T <: Real}(d::Dirichlet, x::Matrix{T})
    r = Array(Float64, size(x, 2))
    logpdf!(r, d, x)
    return r
end

function rand(d::Dirichlet)
    x = [rand(Gamma(el)) for el in d.alpha]
    return x ./ sum(x)
end

function rand!(d::Dirichlet, X::Matrix)
    m, n = size(X)
    for i in 1:n
        X[:, i] = rand(Gamma(d.alpha[i]), m)
    end
    for i in 1:m
        isum = sum(X[i, :])
        for j in 1:n
            X[i, j] /= isum
        end
    end
    return X
end

function var(d::Dirichlet)
    alpha0 = sum(d.alpha)
    return d.alpha .* (alpha0 - d.alpha) / (alpha0^2 * (alpha0 + 1))
end
