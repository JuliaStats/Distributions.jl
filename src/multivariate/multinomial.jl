immutable Multinomial{T<:Real} <: DiscreteMultivariateDistribution
    n::Int
    p::Vector{T}

    function Multinomial(n::Integer, p::Vector{T})
        if n <= 0
            throw(ArgumentError("n must be a positive integer."))
        end
        if !isprobvec(p)
            throw(ArgumentError("p = $p is not a probability vector."))
        end
        new(round(Int, n), p)
    end
    Multinomial(n::Integer, p::Vector{T}, ::NoArgCheck) = new(round(Int, n), p)
end
Multinomial{T<:Real}(n::Integer, p::Vector{T}) = Multinomial{T}(n, p)
Multinomial(n::Integer, k::Integer) = Multinomial{Float64}(round(Int, n), fill(1.0 / k, k))

# Parameters

ncategories(d::Multinomial) = length(d.p)
length(d::Multinomial) = ncategories(d)
probs(d::Multinomial) = d.p
ntrials(d::Multinomial) = d.n

params(d::Multinomial) = (d.n, d.p)
@inline partype{T<:Real}(d::Multinomial{T}) = T

### Conversions
convert{T<:Real}(::Type{Multinomial{T}}, d::Multinomial) = Multinomial(d.n, Vector{T}(d.p))
convert{T<:Real}(::Type{Multinomial{T}}, n, p::Vector) = Multinomial(n, Vector{T}(p))

# Statistics

mean(d::Multinomial) = d.n .* d.p

function var{T<:Real}(d::Multinomial{T})
    p = probs(d)
    k = length(p)
    n = ntrials(d)

    v = Array(T, k)
    for i = 1:k
        @inbounds p_i = p[i]
        v[i] = n * p_i * (1 - p_i)
    end
    v
end

function cov{T<:Real}(d::Multinomial{T})
    p = probs(d)
    k = length(p)
    n = ntrials(d)

    C = Array(T, k, k)
    for j = 1:k
        pj = p[j]
        for i = 1:j-1
            @inbounds C[i,j] = - n * p[i] * pj
        end

        @inbounds C[j,j] = n * pj * (1-pj)
    end

    for j = 1:k-1
        for i = j+1:k
            @inbounds C[i,j] = C[j,i]
        end
    end
    C
end

function mgf{T<:Real}(d::Multinomial{T}, t::AbstractVector)
    p = probs(d)
    n = ntrials(p)
    s = zero(T)
    for i in 1:length(p)
        s += p[i] * exp(t[i])
    end
    return s^n
end

function cf{T<:Real}(d::Multinomial{T}, t::AbstractVector)
    p = probs(d)
    n = ntrials(d)
    s = zero(Complex{T})
    for i in 1:length(p)
        s += p[i] * exp(im * t[i])
    end
    return s^n
end

function entropy(d::Multinomial)
    n, p = params(d)
    s = -lgamma(n+1) + n*entropy(p)
    for pr in p
        b = Binomial(n, pr)
        for x in 0:n
            s += pdf(b, x) * lgamma(x+1)
        end
    end
    return s
end


# Evaluation

function insupport{T<:Real}(d::Multinomial, x::AbstractVector{T})
    k = length(d)
    length(x) == k || return false
    s = 0.0
    for i = 1:k
        @inbounds xi = x[i]
        if !(isinteger(xi) && xi >= 0)
            return false
        end
        s += xi
    end
    return s == ntrials(d)  # integer computation would not yield truncation errors
end

function _logpdf{T<:Real}(d::Multinomial, x::AbstractVector{T})
    p = probs(d)
    n = ntrials(d)
    S = eltype(p)
    R = promote_type(T, S)
    s = R(lgamma(n + 1))
    t = zero(T)
    for i = 1:length(p)
        @inbounds xi = x[i]
        @inbounds p_i = p[i]
        t += xi
        s -= R(lgamma(xi + 1))
        @inbounds s += xi * log(p_i)
    end
    return ifelse(t == n, s, -R(Inf))
end

# Sampling

_rand!{T<:Real}(d::Multinomial, x::AbstractVector{T}) = multinom_rand!(ntrials(d), probs(d), x)

sampler(d::Multinomial) = MultinomialSampler(ntrials(d), probs(d))


## Fit model

immutable MultinomialStats <: SufficientStats
    n::Int  # number of trials in each experiment
    scnts::Vector{Float64}  # sum of counts
    tw::Float64  # total sample weight

    MultinomialStats(n::Int, scnts::Vector{Float64}, tw::Real) = new(n, scnts, Float64(tw))
end

function suffstats{T<:Real}(::Type{Multinomial}, x::Matrix{T})
    K = size(x, 1)
    n::T = zero(T)
    scnts = zeros(K)

    for j = 1:size(x,2)
        nj = zero(T)
        for i = 1:K
            @inbounds xi = x[i,j]
            @inbounds scnts[i] += xi
            nj += xi
        end

        if j == 1
            n = nj
        elseif nj != n
            error("Each sample in X should sum to the same value.")
        end
    end
    MultinomialStats(n, scnts, size(x,2))
end

function suffstats{T<:Real}(::Type{Multinomial}, x::Matrix{T}, w::Array{Float64})
    length(w) == size(x, 2) || throw(ArgumentError("Inconsistent argument dimensions."))

    K = size(x, 1)
    n::T = zero(T)
    scnts = zeros(K)
    tw = 0.

    for j = 1:size(x,2)
        nj = zero(T)
        @inbounds wj = w[j]
        tw += wj
        for i = 1:K
            @inbounds xi = x[i,j]
            @inbounds scnts[i] += xi * wj
            nj += xi
        end

        if j == 1
            n = nj
        elseif nj != n
            error("Each sample in X should sum to the same value.")
        end
    end
    MultinomialStats(n, scnts, tw)
end

fit_mle(::Type{Multinomial}, ss::MultinomialStats) = Multinomial(ss.n, ss.scnts * inv(ss.tw * ss.n))

function fit_mle{T<:Real}(::Type{Multinomial}, x::Matrix{T})
    ss = suffstats(Multinomial, x)
    Multinomial(ss.n, multiply!(ss.scnts, inv(ss.tw * ss.n)))
end

function fit_mle{T<:Real}(::Type{Multinomial}, x::Matrix{T}, w::Array{Float64})
    ss = suffstats(Multinomial, x, w)
    Multinomial(ss.n, multiply!(ss.scnts, inv(ss.tw * ss.n)))
end
