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

`Categorical` is simply a type alias describing a special case of a
`DiscreteNonParametric` distribution, so non-specialized methods defined for
`DiscreteNonParametric` apply to `Categorical` as well.

External links:

* [Categorical distribution on Wikipedia](http://en.wikipedia.org/wiki/Categorical_distribution)
"""
const Categorical{P,Ps} = DiscreteNonParametric{Int,P,Base.OneTo{Int},Ps}

function Categorical{P,Ps}(p::Ps; check_args=true) where {P<:Real, Ps<:AbstractVector{P}}
    check_args && @check_args(Categorical, isprobvec(p))
    return Categorical{P,Ps}(Base.OneTo(length(p)), p, check_args=check_args)
end

Categorical(p::Ps; check_args=true) where {P<:Real, Ps<:AbstractVector{P}} =
    Categorical{P,Ps}(p, check_args=check_args)

function Categorical(k::Integer; check_args=true)
    check_args && @check_args(Categorical, k >= 1)
    return Categorical{Float64,Vector{Float64}}(Base.OneTo(k), fill(1/k, k), check_args=check_args)
end

Categorical(probabilities::Real...; check_args=true) = Categorical([probabilities...]; check_args=check_args)

### Conversions

convert(::Type{Categorical{P,Ps}}, x::AbstractVector{<:Real}) where {
    P<:Real,Ps<:AbstractVector{P}} = Categorical{P,Ps}(Ps(x))

### Parameters

ncategories(d::Categorical) = support(d).stop
params(d::Categorical{P,Ps}) where {P<:Real, Ps<:AbstractVector{P}} = (probs(d),)
partype(::Categorical{T}) where {T<:Real} = T

### Statistics

function median(d::Categorical{T}) where {T<:Real}
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

pdf(d::Categorical{T}, x::Int) where {T<:Real} = insupport(d, x) ? probs(d)[x] : zero(T)

function _pdf!(r::AbstractArray, d::Categorical{T}, rgn::UnitRange) where {T<:Real}
    vfirst = round(Int, first(rgn))
    vlast = round(Int, last(rgn))
    vl = max(vfirst, 1)
    vr = min(vlast, ncategories(d))
    p = probs(d)
    if vl > vfirst
        for i = 1:(vl - vfirst)
            r[i] = zero(T)
        end
    end
    fm1 = vfirst - 1
    for v = vl:vr
        r[v - fm1] = p[v]
    end
    if vr < vlast
        for i = (vr - vfirst + 2):length(rgn)
            r[i] = zero(T)
        end
    end
    return r
end


# sampling

sampler(d::Categorical{P,Ps}) where {P<:Real,Ps<:AbstractVector{P}} =
   AliasTable(probs(d))


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

function suffstats(::Type{<:Categorical}, k::Int, x::AbstractArray{T}) where T<:Integer
    CategoricalStats(add_categorical_counts!(zeros(k), x))
end

function suffstats(::Type{<:Categorical}, k::Int, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    CategoricalStats(add_categorical_counts!(zeros(k), x, w))
end

const CategoricalData = Tuple{Int, AbstractArray}

suffstats(::Type{<:Categorical}, data::CategoricalData) = suffstats(Categorical, data...)
suffstats(::Type{<:Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = suffstats(Categorical, data..., w)

# Model fitting

function fit_mle(::Type{<:Categorical}, ss::CategoricalStats)
    Categorical(pnormalize!(ss.h))
end

function fit_mle(::Type{<:Categorical}, k::Integer, x::AbstractArray{T}) where T<:Integer
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x)), check_args=false)
end

function fit_mle(::Type{<:Categorical}, k::Integer, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x, w)), check_args=false)
end

fit_mle(::Type{<:Categorical}, data::CategoricalData) = fit_mle(Categorical, data...)
fit_mle(::Type{<:Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = fit_mle(Categorical, data..., w)

fit_mle(::Type{<:Categorical}, x::AbstractArray{T}) where {T<:Integer} = fit_mle(Categorical, maximum(x), x)
fit_mle(::Type{<:Categorical}, x::AbstractArray{T}, w::AbstractArray{Float64}) where {T<:Integer} = fit_mle(Categorical, maximum(x), x, w)

fit(::Type{<:Categorical}, data::CategoricalData) = fit_mle(Categorical, data)
fit(::Type{<:Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = fit_mle(Categorical, data, w)
