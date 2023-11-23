"""
The [Multinomial distribution](http://en.wikipedia.org/wiki/Multinomial_distribution)
generalizes the *binomial distribution*. Consider n independent draws from a Categorical
distribution over a finite set of size k, and let ``X = (X_1, ..., X_k)`` where ``X_i``
represents the number of times the element ``i`` occurs, then the distribution of ``X``
is a multinomial distribution. Each sample of a multinomial distribution is a k-dimensional
integer vector that sums to n.

The probability mass function is given by

```math
f(x; n, p) = \\frac{n!}{x_1! \\cdots x_k!} \\prod_{i=1}^k p_i^{x_i},
\\quad x_1 + \\cdots + x_k = n
```

```julia
Multinomial(n, p)   # Multinomial distribution for n trials with probability vector p
Multinomial(n, k)   # Multinomial distribution for n trials with equal probabilities
                    # over 1:k
```
"""
struct Multinomial{T<:Real, TV<:AbstractVector{T}} <: DiscreteMultivariateDistribution
    n::Int
    p::TV
    Multinomial{T, TV}(n::Int, p::TV) where {T <: Real, TV <: AbstractVector{T}} = new{T, TV}(n, p)
end

function Multinomial(n::Integer, p::AbstractVector{T}; check_args::Bool=true) where {T<:Real}
    @check_args(
        Multinomial,
        (n, n >= 0),
        (p, isprobvec(p), "p is not a probability vector."),
    )
    return Multinomial{T,typeof(p)}(n, p)
end

function Multinomial(n::Integer, k::Integer; check_args::Bool=true)
    @check_args Multinomial (n, n >= 0) (k, k >= 1)
    return Multinomial{Float64, Vector{Float64}}(round(Int, n), fill(1.0 / k, k))
end

# Parameters

ncategories(d::Multinomial) = length(d.p)
length(d::Multinomial) = ncategories(d)
probs(d::Multinomial) = d.p
ntrials(d::Multinomial) = d.n

params(d::Multinomial) = (d.n, d.p)
@inline partype(d::Multinomial{T}) where {T<:Real} = T

### Conversions
convert(::Type{Multinomial{T, TV}}, d::Multinomial) where {T<:Real, TV<:AbstractVector{T}} = Multinomial(d.n, TV(d.p))
convert(::Type{Multinomial{T, TV}}, d::Multinomial{T, TV}) where {T<:Real, TV<:AbstractVector{T}} = d
convert(::Type{Multinomial{T, TV}}, n, p::AbstractVector) where {T<:Real, TV<:AbstractVector} = Multinomial(n, TV(p))
convert(::Type{Multinomial{T}}, d::Multinomial) where {T<:Real} = Multinomial(d.n, T.(d.p))
convert(::Type{Multinomial{T}}, d::Multinomial{T}) where {T<:Real} = d
convert(::Type{Multinomial{T}}, n, p::AbstractVector) where {T<:Real} = Multinomial(n, T.(p))

# Statistics

mean(d::Multinomial) = d.n .* d.p

function var(d::Multinomial{T}) where T<:Real
    p = probs(d)
    k = length(p)
    n = ntrials(d)

    v = Vector{T}(undef, k)
    for i = 1:k
        @inbounds p_i = p[i]
        v[i] = n * p_i * (1 - p_i)
    end
    v
end

function cov(d::Multinomial{T}) where T<:Real
    p = probs(d)
    k = length(p)
    n = ntrials(d)

    C = Matrix{T}(undef, k, k)
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

function mgf(d::Multinomial{T}, t::AbstractVector) where T<:Real
    p = probs(d)
    n = ntrials(p)
    s = zero(T)
    for i in 1:length(p)
        s += p[i] * exp(t[i])
    end
    return s^n
end

function cf(d::Multinomial{T}, t::AbstractVector) where T<:Real
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
    s = -loggamma(n+1) + n*entropy(p)
    for pr in p
        b = Binomial(n, pr)
        for x in 0:n
            s += pdf(b, x) * loggamma(x+1)
        end
    end
    return s
end


# Evaluation

function insupport(d::Multinomial, x::AbstractVector{T}) where T<:Real
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

function _logpdf(d::Multinomial, x::AbstractVector{T}) where T<:Real
    p = probs(d)
    n = ntrials(d)
    S = eltype(p)
    R = promote_type(T, S)
    insupport(d,x) || return -R(Inf)
    s = R(loggamma(n + 1))
    for i = 1:length(p)
        @inbounds xi = x[i]
        @inbounds p_i = p[i]
        s -= R(loggamma(R(xi) + 1))
        s += xlogy(xi, p_i)
    end
    return s
end

# Sampling

# if only a single sample is requested, no alias table is created
_rand!(rng::AbstractRNG, d::Multinomial, x::AbstractVector{<:Real}) =
    multinom_rand!(rng, ntrials(d), probs(d), x)

sampler(d::Multinomial) = MultinomialSampler(ntrials(d), probs(d))


## Fit model

struct MultinomialStats <: SufficientStats
    n::Int  # number of trials in each experiment
    scnts::Vector{Float64}  # sum of counts
    tw::Float64  # total sample weight

    MultinomialStats(n::Int, scnts::Vector{Float64}, tw::Real) = new(n, scnts, Float64(tw))
end

function suffstats(::Type{<:Multinomial}, x::Matrix{T}) where T<:Real
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

function suffstats(::Type{<:Multinomial}, x::Matrix{T}, w::Array{Float64}) where T<:Real
    length(w) == size(x, 2) || throw(DimensionMismatch("Inconsistent argument dimensions."))

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

fit_mle(::Type{<:Multinomial}, ss::MultinomialStats) = Multinomial(ss.n, ss.scnts * inv(ss.tw * ss.n))

function fit_mle(::Type{<:Multinomial}, x::Matrix{<:Real})
    ss = suffstats(Multinomial, x)
    Multinomial(ss.n, lmul!(inv(ss.tw * ss.n), ss.scnts))
end

function fit_mle(::Type{<:Multinomial}, x::Matrix{<:Real}, w::Array{Float64})
    ss = suffstats(Multinomial, x, w)
    Multinomial(ss.n, lmul!(inv(ss.tw * ss.n), ss.scnts))
end
