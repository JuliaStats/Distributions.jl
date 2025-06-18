"""
    truncated(d0::UnivariateDistribution; [lower::Real], [upper::Real])
    truncated(d0::UnivariateDistribution, lower::Real, upper::Real)

A _truncated distribution_ `d` of a distribution `d0` to the interval
``[l, u]=```[lower, upper]` has the probability density (mass) function:

```math
f(x; d_0, l, u) = \\frac{f_{d_0}(x)}{P_{Z \\sim d_0}(l \\le Z \\le u)}, \\quad x \\in [l, u],
```
where ``f_{d_0}(x)`` is the probability density (mass) function of ``d_0``.

The function throws an error if ``l > u``.

```julia
truncated(d0; lower=l)           # d0 left-truncated to the interval [l, Inf)
truncated(d0; upper=u)           # d0 right-truncated to the interval (-Inf, u]
truncated(d0; lower=l, upper=u)  # d0 truncated to the interval [l, u]
truncated(d0, l, u)              # d0 truncated to the interval [l, u]
```

The function falls back to constructing a [`Truncated`](@ref) wrapper.

# Implementation

To implement a specialized truncated form for distributions of type `D`, one or more of the
following methods should be implemented:
- `truncated(d0::D, l::T, u::T) where {T <: Real}`: interval-truncated
- `truncated(d0::D, ::Nothing, u::Real)`: right-truncated
- `truncated(d0::D, l::Real, u::Nothing)`: left-truncated
"""
function truncated end
function truncated(d::UnivariateDistribution, l::Real, u::Real)
    return truncated(d, promote(l, u)...)
end
function truncated(
    d::UnivariateDistribution;
    lower::Union{Real,Nothing}=nothing,
    upper::Union{Real,Nothing}=nothing,
)
    return truncated(d, lower, upper)
end
function truncated(d::UnivariateDistribution, ::Nothing, u::Real)
    # (log)ucdf = (log)tp = (log) P(X ≤ u) where X ~ d
    logucdf = logtp = logcdf(d, u)
    ucdf = tp = exp(logucdf)

    Truncated(d, nothing, promote(u, oftype(ucdf, -Inf), zero(ucdf), ucdf, tp, logtp)...)
end
function truncated(d::UnivariateDistribution, l::Real, ::Nothing)
    # (log)lcdf = (log) P(X < l) where X ~ d
    loglcdf = _logcdf_noninclusive(d, l)
    lcdf = exp(loglcdf)

    # (log)tp = (log) P(l ≤ X) where X ∼ d
    logtp = log1mexp(loglcdf)
    tp = exp(logtp)

    l, loglcdf, lcdf, ucdf, tp, logtp = promote(l, loglcdf, lcdf, one(lcdf), tp, logtp)
    Truncated(d, l, nothing, loglcdf, lcdf, ucdf, tp, logtp)
end
truncated(d::UnivariateDistribution, ::Nothing, ::Nothing) = d
function truncated(d::UnivariateDistribution, l::T, u::T) where {T <: Real}
    l <= u || error("the lower bound must be less or equal than the upper bound")

    # (log)lcdf = (log) P(X < l) where X ~ d
    loglcdf = _logcdf_noninclusive(d, l)
    lcdf = exp(loglcdf)

    # (log)ucdf = (log) P(X ≤ u) where X ~ d
    logucdf = logcdf(d, u)
    ucdf = exp(logucdf)

    # (log)tp = (log) P(l ≤ X ≤ u) where X ∼ d
    logtp = logsubexp(loglcdf, logucdf)
    tp = exp(logtp)

    Truncated(d, promote(l, u, loglcdf, lcdf, ucdf, tp, logtp)...)
end

"""
    Truncated

Generic wrapper for a truncated distribution.

The *truncated normal distribution* is a particularly important one in the family of truncated distributions.
Unlike the general case, truncated normal distributions support `mean`, `mode`, `modes`, `var`, `std`, and `entropy`.
"""
struct Truncated{D<:UnivariateDistribution, S<:ValueSupport, T<: Real, TL<:Union{T,Nothing}, TU<:Union{T,Nothing}} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::TL     # lower bound
    upper::TU     # upper bound
    loglcdf::T    # log-cdf of lower bound (exclusive): log P(X < lower)
    lcdf::T       # cdf of lower bound (exclusive): P(X < lower)
    ucdf::T       # cdf of upper bound (inclusive): P(X ≤ upper)

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)

    function Truncated(d::UnivariateDistribution, l::TL, u::TU, loglcdf::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real, TL <: Union{T,Nothing}, TU <: Union{T,Nothing}}
        new{typeof(d), value_support(typeof(d)), T, TL, TU}(d, l, u, loglcdf, lcdf, ucdf, tp, logtp)
    end
end

const LeftTruncated{D<:UnivariateDistribution,S<:ValueSupport,T<:Real} = Truncated{D,S,T,T,Nothing}
const RightTruncated{D<:UnivariateDistribution,S<:ValueSupport,T<:Real} = Truncated{D,S,T,Nothing,T}

### Constructors of `Truncated` are deprecated - users should call `truncated`
@deprecate Truncated(d::UnivariateDistribution, l::Real, u::Real) truncated(d, l, u)
@deprecate Truncated(d::UnivariateDistribution, l::T, u::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real} Truncated(d, l, u, log(lcdf), lcdf, ucdf, tp, logtp)

function truncated(d::Truncated, l::T, u::T) where {T<:Real}
    return truncated(
        d.untruncated,
        d.lower === nothing ? l : max(l, d.lower),
        d.upper === nothing ? u : min(u, d.upper),
    )
end
function truncated(d::Truncated, ::Nothing, u::Real)
    return truncated(d.untruncated, d.lower, d.upper === nothing ? u : min(u, d.upper))
end
function truncated(d::Truncated, l::Real, ::Nothing)
    return truncated(d.untruncated, d.lower === nothing ? l : max(l, d.lower), d.upper)
end

params(d::Truncated) = tuple(params(d.untruncated)..., d.lower, d.upper)
partype(d::Truncated{<:UnivariateDistribution,<:ValueSupport,T}) where {T<:Real} = promote_type(partype(d.untruncated), T)

Base.eltype(::Type{<:Truncated{D}}) where {D<:UnivariateDistribution} = eltype(D)
Base.eltype(d::Truncated) = eltype(d.untruncated)

### range and support

islowerbounded(d::RightTruncated) = islowerbounded(d.untruncated)
islowerbounded(d::Truncated) = islowerbounded(d.untruncated) || isfinite(d.lower)

isupperbounded(d::LeftTruncated) = isupperbounded(d.untruncated)
isupperbounded(d::Truncated) = isupperbounded(d.untruncated) || isfinite(d.upper)

minimum(d::RightTruncated) = minimum(d.untruncated)
minimum(d::Truncated) = max(minimum(d.untruncated), d.lower)

maximum(d::LeftTruncated) = maximum(d.untruncated)
maximum(d::Truncated) = min(maximum(d.untruncated), d.upper)

function insupport(d::Truncated{<:UnivariateDistribution,<:Union{Discrete,Continuous}}, x::Real)
    return _in_closed_interval(x, d.lower, d.upper) && insupport(d.untruncated, x)
end

### evaluation

function quantile(d::Truncated, p::Real)
    x = quantile(d.untruncated, d.lcdf + p * d.tp)
    min_x, max_x = extrema(d)
    return clamp(x, oftype(x, min_x), oftype(x, max_x))
end

function pdf(d::Truncated, x::Real)
    result = pdf(d.untruncated, x) / d.tp
    return _in_closed_interval(x, d.lower, d.upper) ? result : zero(result)
end

function logpdf(d::Truncated, x::Real)
    result = logpdf(d.untruncated, x) - d.logtp
    return _in_closed_interval(x, d.lower, d.upper) ? result : oftype(result, -Inf)
end

function cdf(d::Truncated, x::Real)
    result = clamp((cdf(d.untruncated, x) - d.lcdf) / d.tp, 0, 1)
    # Special cases for values outside of the support to avoid e.g. NaN issues with `Binomial`
    return if d.lower !== nothing && x < d.lower
        zero(result)
    elseif d.upper !== nothing && x >= d.upper
        one(result)
    else
        result
    end
end

function logcdf(d::Truncated, x::Real)
    result = logsubexp(logcdf(d.untruncated, x), d.loglcdf) - d.logtp
    return if d.lower !== nothing && x < d.lower
        oftype(result, -Inf)
    elseif d.upper !== nothing && x >= d.upper
        zero(result)
    else
        result
    end
end

function ccdf(d::Truncated, x::Real)
    result = clamp((d.ucdf - cdf(d.untruncated, x)) / d.tp, 0, 1)
    # Special cases for values outside of the support to avoid e.g. NaN issues with `Binomial`
    return if d.lower !== nothing && x <= d.lower
        one(result)
    elseif d.upper !== nothing && x > d.upper
        zero(result)
    else
        result
    end
end

function logccdf(d::Truncated, x::Real)
    result = logsubexp(logccdf(d.untruncated, x), log1p(-d.ucdf)) - d.logtp
    return if d.lower !== nothing && x <= d.lower
        zero(result)
    elseif d.upper !== nothing && x > d.upper
        oftype(result, -Inf)
    else
        result
    end
end

## random number generation


"""
    rand(rng::AbstractRNG, d::Truncated)

Generate a single random sample from a truncated distribution.

The sampling strategy depends on the probability mass of the truncated region (`tp`):
- If `tp > 0.25`, rejection sampling is used. This is efficient when the truncated region covers a large portion of the original distribution.
- If `sqrt(eps) < tp <= 0.25`, inverse transform sampling is used. This is more efficient for smaller truncated regions.
- If `tp` is very small (`<= sqrt(eps)`), a numerically stable version of inverse transform sampling is used which performs calculations in log-space to maintain precision.
"""
function rand(rng::AbstractRNG, d::Truncated)
    d0 = d.untruncated
    tp = d.tp
    lower = d.lower
    upper = d.upper
    if tp > 0.25
        while true
            r = rand(rng, d0)
            if _in_closed_interval(r, lower, upper)
                return r
            end
        end
    elseif tp > sqrt(eps(typeof(float(tp))))
        return quantile(d0, d.lcdf + rand(rng) * d.tp)
    else
        # computation in log-space fixes numerical issues if d.tp is small (#1548)
        return invlogcdf(d0, logaddexp(d.loglcdf, d.logtp - randexp(rng)))
    end
end


"""
    rand(rng::AbstractRNG, d::Truncated, n::Int)

Generate `n` random samples from a truncated distribution.

The implementation samples the untruncated distribution, `d0` with `rand(rng, d0, n)` in batches and only keeps the samples that fall within the truncated range. The size of the batches is adaptively estimated to reduce the number of iterations.

See [rand(rng::AbstractRNG, d::Truncated)](@ref) that handles the case of small mass of the truncated region.

!!! warning
    This method can be inefficient if the probability mass of the truncated region is very small. 
"""
function rand(rng::AbstractRNG, d::Truncated, n::Int)
    n == 0 && return partype(d)[]
    # 
    d0 = d.untruncated
    tp = d.tp
    lower = d.lower
    upper = d.upper
    # Preallocate samples array
    samples = Vector{partype(d)}(undef, n)
    n_collected = 0
    while n_collected < n
        n_remaining = n - n_collected
        # Estimate number of samples to draw from the untruncated distribution.
        # We draw more to reduce the chance of needing more rounds.
        n_expected = n_remaining / tp
        δn_expected = sqrt(n_remaining * tp * (1 - tp))  # standard deviation of the expected number
        n_batch = ceil(Int, n_expected + 3δn_expected)
        samples_d0 = rand(rng, d0, n_batch)
        for s in samples_d0
            if _in_closed_interval(s, lower, upper)
                n_collected += 1
                samples[n_collected] = s
                n_collected == n && break
            end
        end
    end
    return samples
end

## show

function show(io::IO, d::Truncated)
    print(io, "Truncated(")
    d0 = d.untruncated
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    if d.lower === nothing
        print(io, "; upper=$(d.upper))")
    elseif d.upper === nothing
        print(io, "; lower=$(d.lower))")
    else
        print(io, "; lower=$(d.lower), upper=$(d.upper))")
    end
    uml && println(io)
end

_use_multline_show(d::Truncated) = _use_multline_show(d.untruncated)


### specialized truncated distributions

include(joinpath("truncated", "normal.jl"))
include(joinpath("truncated", "exponential.jl"))
include(joinpath("truncated", "uniform.jl"))
include(joinpath("truncated", "loguniform.jl"))
include(joinpath("truncated", "discrete_uniform.jl"))

#### Utilities

# utilities to handle closed intervals represented with possibly `nothing` bounds
_in_closed_interval(x::Real, l::Real, u::Real) = l ≤ x ≤ u
_in_closed_interval(x::Real, ::Nothing, u::Real) = x ≤ u
_in_closed_interval(x::Real, l::Real, ::Nothing) = x ≥ l
