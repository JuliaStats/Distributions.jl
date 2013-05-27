immutable Multinomial <: DiscreteMultivariateDistribution
    n::Int
    prob::Vector{Float64}
    drawtable::DiscreteDistributionTable
    function Multinomial{T <: Real}(n::Integer, p::Vector{T})
        if n <= 0
            error("Multinomial: n must be positive")
        end
        sump = 0.0
        for i in 1:length(p)
            if p[i] < 0.0
                error("Multinomial: probabilities must be non-negative")
            end
            sump += p[i]
        end
        for i in 1:length(p)
            p[i] /= sump
        end
        new(int(n), p, DiscreteDistributionTable(p))
    end
end

function Multinomial(n::Integer, d::Integer)
    if d <= 1
        error("d must be greater than 1")
    end
    prob = Array(Float64, d)
    fill!(prob, 1.0 / d)
    Multinomial(n, prob)
end

Multinomial(d::Integer) = Multinomial(1, d)

entropy(d::Multinomial) = pventropy(d.prob)

function insupport{T <: Real}(d::Multinomial, x::Vector{T})
    n = length(x)
    if length(d.prob) != n
        return false
    end
    s = 0.0
    for i in 1:n
        if x[i] < 0.0 || !isinteger(x[i])
            return false
        end
        s += x[i]
    end
    if abs(s - d.n) > 1e-8
        return false
    end
    return true
end

mean(d::Multinomial) = d.n .* d.prob

function mgf(d::Multinomial, t::AbstractVector)
    p, n = d.prob, d.n
    k = length(p)
    s = 0.0
    for i in 1:k
        s += p[i] * exp(t[i])
    end
    return s^n
end

function cf(d::Multinomial, t::AbstractVector)
    p, n = d.prob, d.n
    k = length(p)
    s = 0.0 + 0.0im
    for i in 1:k
        s += p[i] * exp(im * t[i])
    end
    return s^n
end

pdf{T <: Real}(d::Multinomial, x::Vector{T}) = exp(logpdf(d, x))

function logpdf{T <: Real}(d::Multinomial, x::Vector{T})
    if !insupport(d, x)
        return -Inf
    else
        s = lgamma(d.n + 1.0)
        for i in 1:length(x)
            s -= lgamma(x[i] + 1.0)
            s += x[i] * log(d.prob[i])
        end
        return s
    end
end

function rand!(d::Multinomial, x::Vector)
    for itr in 1:d.n
        i = draw(d.drawtable)
        x[i] += 1
    end
    return x
end

function rand(d::Multinomial)
    x = zeros(Int, length(d.prob))
    return rand!(d, x)
end

function var(d::Multinomial)
    n = length(d.prob)
    S = Array(Float64, n, n)
    for j in 1:n
        for i in 1:n
            if i == j
                S[i, j] = d.n * d.prob[i] * (1.0 - d.prob[i])
            else
                S[i, j] = -d.n * d.prob[i] * d.prob[j]
            end
        end
    end
    return S
end
