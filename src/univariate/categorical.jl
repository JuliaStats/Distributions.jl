immutable Categorical <: DiscreteUnivariateDistribution
    K::Int
    prob::Vector{Float64}

    Categorical(p::Vector{Float64}, ::NoArgCheck) = new(length(p), p)

    function Categorical(p::Vector{Float64})
        isprobvec(p) || error("p is not a valid probability vector.")
        new(length(p), p)
    end

    Categorical(k::Int) = new(k, fill(1.0/k, k))
end

### handling support

function insupport(d::Categorical, x::Real)
    isinteger(x) && one(x) <= x <= d.K && d.prob[x] != 0.0
end

isupperbounded(::Union(Categorical, Type{Categorical})) = true
islowerbounded(::Union(Categorical, Type{Categorical})) = true
isbounded(::Union(Categorical, Type{Categorical})) = true

minimum(::Union(Categorical, Type{Categorical})) = 1
maximum(d::Categorical) = d.K
support(d::Categorical) = 1:d.K

# evaluation

function cdf(d::Categorical, x::Real)
    x < one(x) && return 0.0
    d.K <= x && return 1.0
    p = d.prob[1]
    for i in 2:ifloor(x)
        p += d.prob[i]
    end
    p
end

pdf(d::Categorical, x::Real) = isinteger(x) && 1 <= x <= d.K ? d.prob[x] : 0.0

logpdf(d::Categorical, x::Real) = isinteger(x) && 1 <= x <= d.K ? log(d.prob[x]) : -Inf

function quantile(d::Categorical, p::Real)
    0. <= p <= 1. || throw(DomainError())
    k = d.K
    pv = d.prob
    i = 1
    v = pv[1]
    while v < p && i < k
        i += 1
        @inbounds v += pv[i]
    end
    i
end

# properties

function categorical_mean(p::AbstractArray{Float64})
    k = length(p)
    s = 0.
    for i = 1 : k
        @inbounds s += p[i] * i
    end
    s
end

mean(d::Categorical) = categorical_mean(d.prob)

function median(d::Categorical)
    k = d.K
    p = d.prob
    cp = 0.
    i = 0
    while cp < 0.5 && i <= k
        i += 1
        @inbounds cp += p[i]
    end
    i
end

function var(d::Categorical)
    k = d.K
    p = d.prob
    m = categorical_mean(p)
    s = 0.0
    for i = 1 : k
        @inbounds s += abs2(i - m) * p[i]
    end
    s
end

function skewness(d::Categorical)
    k = d.K
    p = d.prob
    m = categorical_mean(p)
    s = 0.0
    for i = 1 : k
        @inbounds s += (i - m)^3 * p[i]
    end
    v = var(d)
    s / (v * sqrt(v))
end

function kurtosis(d::Categorical)
    k = d.K
    p = d.prob
    m = categorical_mean(p)
    s = 0.0
    for i = 1 : k
        @inbounds s += (i - m)^4 * p[i]
    end
    s / abs2(var(d)) - 3.0
end

entropy(d::Categorical) = entropy(d.prob)

function mgf(d::Categorical, t::AbstractVector)
    k = d.K
    p = d.prob
    s = 0.0
    for i = 1 : k
        @inbounds s += p[i] * exp(t[i])
    end
    s
end

function cf(d::Categorical, t::AbstractVector)
    k = d.K
    p = d.prob
    s = 0.0 + 0.0im
    for i = 1 : k
        @inbounds s += p[i] * exp(im * t[i])
    end
    s
end

mode(d::Categorical) = indmax(d.prob)

function modes(d::Categorical)
    K = d.K
    p = d.prob
    maxp = maximum(p)
    r = Array(Int, 0)
    for k = 1:K
        @inbounds if p[k] == maxp
            push!(r, k)
        end
    end
    r
end


# sampling

immutable CategoricalSampler
    d::Categorical
    alias::AliasTable
    function CategoricalSampler(d::Categorical)
        new(d, AliasTable(d.prob))
    end
end

sampler(d::Categorical) = CategoricalSampler(d)

rand(d::Categorical) = sample(WeightVec(d.prob, 1.0))

rand(s::CategoricalSampler) = rand(s.alias)


### sufficient statistics

immutable CategoricalStats <: SufficientStats
    h::Vector{Float64}
end

function add_categorical_counts!{T<:Integer}(h::Vector{Float64}, x::AbstractArray{T})
    for i = 1 : length(x)
        @inbounds xi = x[i]
        h[xi] += 1.   # cannot use @inbounds, as no guarantee that x[i] is in bound 
    end
    h
end

function add_categorical_counts!{T<:Integer}(h::Vector{Float64}, x::AbstractArray{T}, w::AbstractArray{Float64})
    n = length(x)
    if n != length(w)
        throw(ArgumentError("Inconsistent array lengths."))
    end
    for i = 1 : n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        h[xi] += wi   # cannot use @inbounds, as no guarantee that x[i] is in bound 
    end
    h
end

function suffstats{T<:Integer}(::Type{Categorical}, k::Int, x::AbstractArray{T})
    CategoricalStats(add_categorical_counts!(zeros(k), x))
end

function suffstats{T<:Integer}(::Type{Categorical}, k::Int, x::AbstractArray{T}, w::AbstractArray{Float64})
    CategoricalStats(add_categorical_counts!(zeros(k), x, w))
end

suffstats{T<:Integer}(::Type{Categorical}, data::(Int, Array{T})) = suffstats(Categorical, data...)
suffstats{T<:Integer}(::Type{Categorical}, data::(Int, Array{T}), w::Array{Float64}) = suffstats(Categorical, data..., w)


### Model fitting

function fit_mle(::Type{Categorical}, ss::CategoricalStats)
    Categorical(pnormalize!(ss.h))
end

function fit_mle{T<:Integer}(::Type{Categorical}, k::Integer, x::Array{T}) 
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x)), NoArgCheck())
end

function fit_mle{T<:Integer}(::Type{Categorical}, k::Integer, x::Array{T}, w::Array{Float64}) 
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x, w)), NoArgCheck())
end

fit_mle{T<:Integer}(::Type{Categorical}, data::(Int, Array{T})) = fit_mle(Categorical, data...)
fit_mle{T<:Integer}(::Type{Categorical}, data::(Int, Array{T}), w::Array{Float64}) = fit_mle(Categorical, data..., w)

fit_mle{T<:Integer}(::Type{Categorical}, x::Array{T}) = fit_mle(Categorical, maximum(x), x)
fit_mle{T<:Integer}(::Type{Categorical}, x::Array{T}, w::Array{Float64}) = fit_mle(Categorical, maximum(x), x, w)


