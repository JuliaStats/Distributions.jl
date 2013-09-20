immutable Categorical <: DiscreteUnivariateDistribution
    K::Int
    prob::Vector{Float64}

    function Categorical(p::Vector{Float64})
        if !isprobvec(p)
            throw(ArgumentError("p = $p is not a probability vector."))
        end
        new(length(p), p)
    end
end

immutable CategoricalSampler <: DiscreteUnivariateDistribution
    d::Categorical
    alias::AliasTable
    function CategoricalSampler(d::Categorical)
        new(d, AliasTable(d.prob))
    end
end

sampler(d::Categorical) = CategoricalSampler(d)

Categorical(d::Integer) = Categorical(ones(d))

min(d::Categorical) = 1
max(d::Categorical) = d.K

function cdf(d::Categorical, x::Real)
    x < one(x) && return 0.0
    d.K <= x && return 1.0
    p = d.prob[1]
    for i in 2:ifloor(x)
        p += d.prob[i]
    end
    p
end

entropy(d::Categorical) = NumericExtensions.entropy(d.prob)

function kurtosis(d::Categorical)
    m = mean(d)
    s = 0.0
    for i in 1:d.K
        s += (i - m)^4 * d.prob[i]
    end
    s / var(d)^2 - 3.0
end

mean(d::Categorical) = sum(Multiply(), [1:d.K], d.prob)

function median(d::Categorical)
    p = 0.
    n = d.K
    i = 0
    while p < 0.5 && i <= n
        i += 1
        p += d.prob[i]
    end
    i
end

function mgf(d::Categorical, t::AbstractVector)
    s = 0.0
    for i in 1:d.K
        s += d.prob[i] * exp(t[i])
    end
    s
end

function cf(d::Categorical, t::AbstractVector)
    s = 0.0 + 0.0im
    for i in 1:d.K
        s += d.prob[i] * exp(im * t[i])
    end
    s
end

mode(d::Categorical) = indmax(d.prob)

function modes(d::Categorical)
    K = d.K
    p = d.prob
    maxp = max(p)
    r = Array(Int, 0)
    for k = 1:K
        if p[k] == maxp
            push!(r, k)
        end
    end
    r
end


pdf(d::Categorical, x::Real) = isinteger(x) && one(x) <= x <= d.K ? d.prob[x] : 0.0

function quantile(d::Categorical, p::Real)
    zero(p) <= p <= one(p) || throw(DomainError())
    k = d.K
    pv = d.prob
    i = 1
    v = pv[1]
    while v < p && i < k
        i += 1
        v += pv[i]
    end
    i
end

function rand(d::Categorical)
    u = rand()
    sump = 0.0 
    for i in 1:d.K
        sump += d.prob[i]
        if u <= sump
            return i
        end
    end
    d.K
end

rand(s::CategoricalSampler) = rand(s.alias)

function skewness(d::Categorical)
    m = mean(d)
    s = 0.0
    for i in 1:d.K
        s += (i - m)^3 * d.prob[i]
    end
    s / std(d)^3
end

function var(d::Categorical)
    m = mean(d)
    s = 0.0
    for i in 1:d.K
        s += (i - m)^2 * d.prob[i]
    end
    s
end

### handling support

function insupport(d::Categorical, x::Real)
    isinteger(x) && one(x) <= x <= d.K && d.prob[x] != 0.0
end

isupperbounded(::Union(Categorical, Type{Categorical})) = true
islowerbounded(::Union(Categorical, Type{Categorical})) = true
isbounded(::Union(Categorical, Type{Categorical})) = true

hasfinitesupport(::Union(Categorical, Type{Categorical})) = true
min(::Union(Categorical, Type{Categorical})) = 1
max(d::Categorical) = d.K
support(d::Categorical) = 1:d.K


### Model fitting

function fit_categorical!{T<:Integer}(p::Vector{Float64}, x::Array{T}; pricount::Float64=0.0)
    k = length(p)
    n = length(x)

    # accumulate counts
    fill!(p, pricount)
    for i = 1:n
        @inbounds p[x[i]] += 1.
    end

    # normalize
    tw = n + pricount * k
    c = 1.0 / tw
    for i = 1:k
        @inbounds p[i] *= c
    end
    return p
end

function fit_categorical!{T<:Integer}(p::Vector{Float64}, x::Array{T}, w::Array{Float64}; pricount::Float64=0.0)
    k = length(p)
    n = length(x)

    # accumulate counts
    fill!(p, pricount)
    tw = pricount * k
    for i = 1:n
        @inbounds wi = w[i]
        @inbounds p[x[i]] += wi 
        tw += wi
    end

    # normalize
    c = 1.0 / tw
    for i = 1:k
        @inbounds p[i] *= c
    end
    return p
end

function fit_categorical{T<:Integer}(k::Integer, x::Array{T}; pricount::Float64=0.0)
    Categorical(fit_categorical!(zeros(k), x; pricount=pricount))
end

function fit_categorical{T<:Integer}(k::Integer, x::Array{T}, w::Array{Float64}; pricount::Float64=0.0)
    Categorical(fit_categorical!(zeros(k), x; pricount=pricount))
end

fit_mle{T<:Integer}(::Type{Categorical}, data::(Int, Array{T})) = fit_categorical(data...)
fit_mle{T<:Integer}(::Type{Categorical}, data::(Int, Array{T}), w::Array{Float64}) = fit_categorical(data..., w)
fit_mle{T<:Integer}(::Type{Categorical}, k::Integer, x::Array{T}) = fit_categorical(k, x)
fit_mle{T<:Integer}(::Type{Categorical}, k::Integer, x::Array{T}, w::Array{Float64}) = fit_categorical(k, x, w)
fit_mle{T<:Integer}(::Type{Categorical}, x::Array{T}) = fit_categorical(max(x), x)
fit_mle{T<:Integer}(::Type{Categorical}, x::Array{T}, w::Array{Float64}) = fit_categorical(max(x), x, w)


