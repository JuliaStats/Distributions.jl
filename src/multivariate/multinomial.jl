immutable Multinomial <: DiscreteMultivariateDistribution
    n::Int
    prob::Vector{Float64}
    function Multinomial{T <: Real}(n::Integer, p::Vector{T})
        p = float(p)
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
        new(int(n), p)
    end
end

immutable MultinomialSampler <: DiscreteMultivariateDistribution
    d::Multinomial
    alias::AliasTable
    function MultinomialSampler(d::Multinomial)
        new(d, AliasTable(d.prob))
    end
end

sampler(d::Multinomial) = MultinomialSampler(d)

function Multinomial(n::Integer, d::Integer)
    if d < 1
        error("d must be greater than 0")
    end
    prob = Array(Float64, d)
    fill!(prob, 1.0 / d)
    Multinomial(n, prob)
end

# TODO: Debate removing this
Multinomial(d::Integer) = Multinomial(1, d)

dim(d::Multinomial) = length(d.prob)
dim(s::MultinomialSampler) = length(s.d.prob)

entropy(d::Multinomial) = NumericExtensions.entropy(d.prob)

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

# TODO: Debate making T <: Integer
function rand!{T <: Real}(d::Multinomial, x::Vector{T})
    n, k = d.n, dim(d)
    fill!(x, 0)
    psum = 1.0
    for j in 1:(k - 1)
        tmp = rand(Binomial(n, d.prob[j] / psum))
        x[j] = tmp
        n -= tmp
        if n == 0
            break
        end
        psum -= d.prob[j]
    end
    x[k] = n
    return x
end

function rand!{T <: Real}(s::MultinomialSampler, x::Vector{T})
    d = s.d
    n, k = d.n, dim(d)
    fill!(x, 0)
    if n^2 > k
        # Use sequential binomial sampling
        # TODO: Refactor this code to make it DRYer
        psum = 1.0
        for j in 1:(k - 1)
            tmp = rand(Binomial(n, d.prob[j] / psum))
            x[j] = tmp
            n -= tmp
            if n == 0
                break
            end
            psum -= d.prob[j]
        end
        x[k] = n
    else
        # Use an alias table
        for itr in 1:n
            x[rand(s.alias)] += 1
        end
    end
    return x
end

function rand(d::Multinomial)
    x = zeros(Int, dim(d))
    return rand!(d, x)
end

function rand(s::MultinomialSampler)
    x = zeros(Int, dim(s.d))
    return rand!(s, x)
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

function fit_mle{T<:Real}(::Type{Multinomial}, X::Matrix{T})
    ns = vec(sum(X, 1))
    if !(all(ns .== ns[1]))
        error("Each sample in X should have the same number of trials.")
    end
    n = int(ns[1])
    p = vec(mean(X, 2)) * (1.0 / n)
    Multinomial(n, p)
end
