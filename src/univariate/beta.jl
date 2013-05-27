immutable Beta <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    function Beta(a::Real, b::Real)
        if a > 0.0 && b > 0.0
            new(float64(a), float64(b))
        else
            error("Both alpha and beta must be positive")
        end
    end
end

Beta(a::Real) = Beta(a, a) # symmetric in [0, 1]
Beta() = Beta(1.0) # uniform

@_jl_dist_2p Beta beta

function entropy(d::Beta)
    o = lbeta(d.alpha, d.beta)
    o = o - (d.alpha - 1.0) * digamma(d.alpha)
    o = o - (d.beta - 1.0) * digamma(d.beta)
    o = o + (d.alpha + d.beta - 2.0) * digamma(d.alpha + d.beta)
    return o
end

insupport(d::Beta, x::Number) = isreal(x) && 0.0 < x < 1.0

function kurtosis(d::Beta)
    a, b = d.alpha, d.beta
    d = 6.0 * ((a - b)^2 * (a + b + 1.0) - a * b * (a + b + 2.0))
    n = a * b * (a + b + 2.0) * (a + b + 3.0)
    return d / n
end

mean(d::Beta) = d.alpha / (d.alpha + d.beta)

function modes(d::Beta)
    if d.alpha > 1.0 && d.beta > 1.0
        return [(d.alpha - 1.0) / (d.alpha + d.beta - 2.0)]
    else
        error("Beta distribution with a <= 1 || b <= 1 has no modes")
    end
end

function rand(d::Beta)
    u = rand(Gamma(d.alpha))
    return u / (u + rand(Gamma(d.beta)))
end

function rand(d::Beta, dims::Dims)
    u = rand(Gamma(d.alpha), dims)
    return u ./ (u + rand(Gamma(d.beta), dims))
end

function rand!(d::Beta, A::Array{Float64})
    A[:] = randbeta(d.alpha, d.beta, size(A))
    return A
end

function skewness(d::Beta)
    d = 2.0 * (d.beta - d.alpha) * sqrt(d.alpha + d.beta + 1.0)
    n = (d.alpha + d.beta + 2.0) * sqrt(d.alpha * d.beta)
    return d / n
end

function var(d::Beta)
    ab = d.alpha + d.beta
    return d.alpha * d.beta / (ab * ab * (ab + 1.0))
end
