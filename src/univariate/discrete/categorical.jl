"""
    Categorical(p)
A *Categorical distribution* is parameterized by a probability vector `p` (of length `K`).

```math
P(X = k) = p[k]  \\quad \\text{for } k = 1, 2, \\ldots, K.
```

```julia
Categorical(p)   # Categorical distribution with probability vector p
params(d)        # Get the parameters, i.e. (p,)
probs(d)         # Get the probability vector, i.e. p
ncategories(d)   # Get the number of categories, i.e. K
```
Here, `p` must be a real vector, of which all components are nonnegative and sum to one.
**Note:** The input vector `p` is directly used as a field of the constructed distribution, without being copied.
External links:
* [Categorical distribution on Wikipedia](http://en.wikipedia.org/wiki/Categorical_distribution)
"""
struct Categorical{T<:Real} <: DiscreteUnivariateDistribution
    K::Int
    p::Vector{T}

    Categorical{T}(p::Vector{T}, ::NoArgCheck) where {T} = new{T}(length(p), p)

    function Categorical{T}(p::Vector{T}) where T
        @check_args(Categorical, isprobvec(p))
        new{T}(length(p), p)
    end

    function Categorical{T}(k::Integer) where T
        @check_args(Categorical, k >= 1)
        new{T}(k, fill(1/k, k))
    end
end

Categorical(p::Vector{T}, ::NoArgCheck) where {T<:Real} = Categorical{T}(p, NoArgCheck())
Categorical(p::Vector{T}) where {T<:Real} = Categorical{T}(p)
Categorical(k::Integer) = Categorical{Float64}(k)

@distr_support Categorical 1 d.K

### Conversions

convert(::Type{Categorical{T}}, p::Vector{S}) where {T<:Real, S<:Real} = Categorical(Vector{T}(p))
convert(::Type{Categorical{T}}, d::Categorical{S}) where {T<:Real, S<:Real} = Categorical(Vector{T}(d.p))

### Parameters

ncategories(d::Categorical) = d.K
probs(d::Categorical) = d.p
params(d::Categorical) = (d.p,)
@inline partype(d::Categorical{T}) where {T<:Real} = T


### Statistics

function categorical_mean(p::AbstractArray{T}) where T<:Real
    k = length(p)
    s = zero(T)
    for i = 1:k
        @inbounds s += p[i] * i
    end
    s
end

mean(d::Categorical) = categorical_mean(d.p)

function median(d::Categorical{T}) where T<:Real
    k = ncategories(d)
    p = probs(d)
    cp = zero(T)
    i = 0
    while cp < 1/2 && i <= k
        i += 1
        @inbounds cp += p[i]
    end
    i
end

function var(d::Categorical{T}) where T<:Real
    k = ncategories(d)
    p = probs(d)
    m = categorical_mean(p)
    s = zero(T)
    for i = 1:k
        @inbounds s += abs2(i - m) * p[i]
    end
    s
end

function skewness(d::Categorical{T}) where T<:Real
    k = ncategories(d)
    p = probs(d)
    m = categorical_mean(p)
    s = zero(T)
    for i = 1:k
        @inbounds s += (i - m)^3 * p[i]
    end
    v = var(d)
    s / (v * sqrt(v))
end

function kurtosis(d::Categorical{T}) where T<:Real
    k = ncategories(d)
    p = probs(d)
    m = categorical_mean(p)
    s = zero(T)
    for i = 1:k
        @inbounds s += (i - m)^4 * p[i]
    end
    s / abs2(var(d)) - 3
end

entropy(d::Categorical) = entropy(d.p)

function mgf(d::Categorical{T}, t::Real) where T<:Real
    k = ncategories(d)
    p = probs(d)
    s = zero(T)
    for i = 1:k
        @inbounds s += p[i] * exp(t)
    end
    s
end

function cf(d::Categorical{T}, t::Real) where T<:Real
    k = ncategories(d)
    p = probs(d)
    s = zero(T) + zero(T)*im
    for i = 1:k
        @inbounds s += p[i] * cis(t)
    end
    s
end

mode(d::Categorical) = argmax(probs(d))

function modes(d::Categorical)
    K = ncategories(d)
    p = probs(d)
    maxp = maximum(p)
    r = Vector{Int}()
    for k = 1:K
        @inbounds if p[k] == maxp
            push!(r, k)
        end
    end
    r
end


### Evaluation

function cdf(d::Categorical{T}, x::Int) where T<:Real
    k = ncategories(d)
    p = probs(d)
    x < 1 && return zero(T)
    x >= k && return one(T)
    c = p[1]
    for i = 2:x
        @inbounds c += p[i]
    end
    return c
end

pdf(d::Categorical{T}, x::Int) where {T<:Real} = insupport(d, x) ? d.p[x] : zero(T)

logpdf(d::Categorical, x::Int) = insupport(d, x) ? log(d.p[x]) : -Inf

function quantile(d::Categorical, p::Float64)
    0 <= p <= 1 || throw(DomainError())
    k = ncategories(d)
    pv = probs(d)
    i = 1
    v = pv[1]
    while v < p && i < k
        i += 1
        @inbounds v += pv[i]
    end
    i
end


# sampling

sampler(d::Categorical) = AliasTable(d.p)


### sufficient statistics

struct CategoricalStats <: SufficientStats
    h::Vector{Float64}
end

function add_categorical_counts!(h::Vector{Float64}, x::AbstractArray{T}) where T<:Integer
    for i = 1 : length(x)
        @inbounds xi = x[i]
        h[xi] += 1.   # cannot use @inbounds, as no guarantee that x[i] is in bound
    end
    h
end

function add_categorical_counts!(h::Vector{Float64}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    n = length(x)
    if n != length(w)
        throw(DimensionMismatch("Inconsistent array lengths."))
    end
    for i = 1 : n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        h[xi] += wi   # cannot use @inbounds, as no guarantee that x[i] is in bound
    end
    h
end

function suffstats(::Type{Categorical}, k::Int, x::AbstractArray{T}) where T<:Integer
    CategoricalStats(add_categorical_counts!(zeros(k), x))
end

function suffstats(::Type{Categorical}, k::Int, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    CategoricalStats(add_categorical_counts!(zeros(k), x, w))
end

const CategoricalData = Tuple{Int, AbstractArray}

suffstats(::Type{Categorical}, data::CategoricalData) = suffstats(Categorical, data...)
suffstats(::Type{Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = suffstats(Categorical, data..., w)

# Model fitting

function fit_mle(::Type{Categorical}, ss::CategoricalStats)
    Categorical(pnormalize!(ss.h))
end

function fit_mle(::Type{Categorical}, k::Integer, x::AbstractArray{T}) where T<:Integer
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x)), NoArgCheck())
end

function fit_mle(::Type{Categorical}, k::Integer, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x, w)), NoArgCheck())
end

fit_mle(::Type{Categorical}, data::CategoricalData) = fit_mle(Categorical, data...)
fit_mle(::Type{Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = fit_mle(Categorical, data..., w)

fit_mle(::Type{Categorical}, x::AbstractArray{T}) where {T<:Integer} = fit_mle(Categorical, maximum(x), x)
fit_mle(::Type{Categorical}, x::AbstractArray{T}, w::AbstractArray{Float64}) where {T<:Integer} = fit_mle(Categorical, maximum(x), x, w)

fit(::Type{Categorical}, data::CategoricalData) = fit_mle(Categorical, data)
fit(::Type{Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = fit_mle(Categorical, data, w)
