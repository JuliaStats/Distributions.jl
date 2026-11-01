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
    # Above the median, 1 - P(X < l) loses the mass in the tail (`loglcdf` rounds to zero), so
    # P(l ≤ X) is evaluated directly
    logtp = loglcdf < -logtwo ? log1mexp(loglcdf) : _logccdf_inclusive(d, l)
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
    # Above the median, P(X ≤ u) - P(X < l) loses the mass in the tail, so it is evaluated as
    # P(X ≥ l) - P(X > u)
    logtp = if loglcdf < -logtwo
        logsubexp(loglcdf, logucdf)
    else
        logsubexp(_logccdf_inclusive(d, l), logccdf(d, u))
    end
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

# Far in a tail, the cached probabilities of the bounds no longer carry the information the
# distribution functions need: above the median, `lcdf` and `ucdf` round to one, and far below
# it, `lcdf`, `ucdf` and `tp` underflow. For a bound above the median, the methods below
# therefore work with the upper tail P(X > x) of the untruncated distribution instead of
# P(X ≤ x), recomputing the tail probability of that bound. The log probabilities are used by
# the log functions and the quantiles, and by `cdf` and `ccdf` where `tp` underflowed.

# whether the lower bound lies above the median, i.e. P(X < lower) ≥ 1/2 (the test `truncated`
# uses to compute `logtp`), and whether the upper bound lies below it, i.e. P(X ≤ upper) ≤ 1/2
_lower_above_median(d::Truncated) = d.loglcdf >= -logtwo
_lower_above_median(::RightTruncated) = false
_upper_below_median(d::Truncated) = 2 * d.ucdf <= 1
_upper_below_median(::LeftTruncated) = false

# log P(X ≥ lower), log P(X ≤ upper) and log P(X > upper) for X ~ d.untruncated, with the
# values of a missing bound
_logccdf_lower(d::Truncated) = _logccdf_inclusive(d.untruncated, d.lower)
_logccdf_lower(d::RightTruncated) = zero(d.logtp)
_logcdf_upper(d::Truncated) = logcdf(d.untruncated, d.upper)
_logcdf_upper(d::LeftTruncated) = zero(d.logtp)
_logccdf_upper(d::Truncated) = logccdf(d.untruncated, d.upper)
_logccdf_upper(d::LeftTruncated) = oftype(d.logtp, -Inf)

# log(1 - exp(a - b)) for log probabilities a ≤ b (up to rounding), i.e. the log of the
# complement of their ratio; log(0) = -Inf for a == b, also when both are -Inf.
# The differences of the (possibly large) log probabilities are taken before `log1mexp`, so
# that a result close to zero keeps its precision and the complement of the probability stays
# accurate.
function _log1mexp_ratio(a::Real, b::Real)
    a == b && return oftype(a - b, -Inf)
    return log1mexp(min(a - b, zero(a - b)))
end

# clamp `x` to the support, converting a bound to the type of `x` only where it replaces `x`
# (the bound of a discrete distribution may be infinite or lie between two support points)
function _clamp_to_support(d::Truncated, x)
    min_x, max_x = extrema(d)
    return x < min_x ? oftype(x, min_x) : x > max_x ? oftype(x, max_x) : x
end

# The inverses of P(X ≤ x) = exp(lp) and of P(X > x) = exp(lp) for X ~ d0. The probability
# itself is passed on where it is a normal floating point number, and its logarithm where it
# underflows. For other number types, such as dual numbers, the probability is always passed
# on: the inverses of the probabilities support more of them than those of the logarithms.
function _invlogcdf(d0::UnivariateDistribution, lp::AbstractFloat)
    p = exp(lp)
    return p >= floatmin(p) ? quantile(d0, p) : invlogcdf(d0, lp)
end
_invlogcdf(d0::UnivariateDistribution, lp::Real) = quantile(d0, exp(lp))
function _invlogccdf(d0::UnivariateDistribution, lp::AbstractFloat)
    p = exp(lp)
    return p >= floatmin(p) ? cquantile(d0, p) : invlogccdf(d0, lp)
end
_invlogccdf(d0::UnivariateDistribution, lp::Real) = cquantile(d0, exp(lp))

# The quantile of `d` for the probability p given as `logp = log(p)` and `log1mp = log(1 - p)`.
# Near the upper end, P(X ≤ x) may round to one while P(X > x) is still accurate, and the other
# way round near the lower end, so the quantile is taken from the tail it is closer to.
function _quantile(d::Truncated, logp::Real, log1mp::Real)
    d0 = d.untruncated
    x = if _lower_above_median(d) || (!_upper_below_median(d) && logp >= -logtwo)
        # P(X > x) = P(X > upper) + (1 - p) P(lower ≤ X ≤ upper)
        _invlogccdf(d0, logaddexp(_logccdf_upper(d), log1mp + d.logtp))
    else
        # P(X ≤ x) = P(X < lower) + p P(lower ≤ X ≤ upper)
        _invlogcdf(d0, logaddexp(d.loglcdf, logp + d.logtp))
    end
    return _clamp_to_support(d, x)
end

# the probabilities are promoted with `logtp` first, so that e.g. a `Float32` probability does
# not limit the precision of a `Float64` distribution's quantile
function quantile(d::Truncated, p::Real)
    p, _ = promote(p, d.logtp)
    return _quantile(d, log(p), log1p(-p))
end
function cquantile(d::Truncated, q::Real)
    q, _ = promote(q, d.logtp)
    return _quantile(d, log1p(-q), log(q))
end
function invlogcdf(d::Truncated, lp::Real)
    lp, _ = promote(lp, d.logtp)
    return _quantile(d, lp, log1mexp(lp))
end
function invlogccdf(d::Truncated, lq::Real)
    lq, _ = promote(lq, d.logtp)
    return _quantile(d, log1mexp(lq), lq)
end

pdf(d::Truncated, x::Real) = exp(logpdf(d, x))

function logpdf(d::Truncated, x::Real)
    result = logpdf(d.untruncated, x) - d.logtp
    return _in_closed_interval(x, d.lower, d.upper) ? result : oftype(result, -Inf)
end

# whether P(lower ≤ X ≤ upper) underflowed, so that only the log probabilities carry information
_tp_underflows(d::Truncated) = !(d.tp >= floatmin(typeof(d.tp)))
# P(X ≥ lower) and P(X > upper) for X ~ d.untruncated, with the values of a missing bound
_ccdf_lower(d::Truncated) = _ccdf_inclusive(d.untruncated, d.lower)
_ccdf_lower(d::RightTruncated) = one(d.tp)
_ccdf_upper(d::Truncated) = ccdf(d.untruncated, d.upper)
_ccdf_upper(d::LeftTruncated) = zero(d.tp)

function cdf(d::Truncated, x::Real)
    if _tp_underflows(d)
        result = exp(logcdf(d, x))
        return min(result, one(result))
    end
    d0 = d.untruncated
    result = if _lower_above_median(d)
        # P(lower ≤ X ≤ x) = P(X ≥ lower) - P(X > x)
        (_ccdf_lower(d) - ccdf(d0, x)) / d.tp
    else
        # P(lower ≤ X ≤ x) = P(X ≤ x) - P(X < lower)
        (cdf(d0, x) - d.lcdf) / d.tp
    end
    result = clamp(result, 0, 1)
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
    d0 = d.untruncated
    result = if _lower_above_median(d)
        # P(lower ≤ X ≤ x) / P(lower ≤ X ≤ upper)
        #   = (1 - P(X > x) / P(X ≥ lower)) / (1 - P(X > upper) / P(X ≥ lower))
        logccdf_lower = _logccdf_lower(d)
        _log1mexp_ratio(logccdf(d0, x), logccdf_lower) -
            _log1mexp_ratio(_logccdf_upper(d), logccdf_lower)
    else
        # P(lower ≤ X ≤ x) / P(lower ≤ X ≤ upper)
        #   = P(X ≤ x) (1 - P(X < lower) / P(X ≤ x)) / P(lower ≤ X ≤ upper)
        logcdf_x = logcdf(d0, x)
        (logcdf_x - d.logtp) + _log1mexp_ratio(d.loglcdf, logcdf_x)
    end
    # Special cases for values outside of the support to avoid e.g. NaN issues with `Binomial`
    return if d.lower !== nothing && x < d.lower
        oftype(result, -Inf)
    elseif d.upper !== nothing && x >= d.upper
        zero(result)
    else
        result
    end
end

function ccdf(d::Truncated, x::Real)
    if _tp_underflows(d)
        result = exp(logccdf(d, x))
        return min(result, one(result))
    end
    d0 = d.untruncated
    result = if _upper_below_median(d)
        # P(x < X ≤ upper) = P(X ≤ upper) - P(X ≤ x)
        (d.ucdf - cdf(d0, x)) / d.tp
    else
        # P(x < X ≤ upper) = P(X > x) - P(X > upper)
        (ccdf(d0, x) - _ccdf_upper(d)) / d.tp
    end
    result = clamp(result, 0, 1)
    # Special cases for values outside of the support to avoid e.g. NaN issues with `Binomial`
    # (at `x == lower` the result is P(X > lower | lower ≤ X ≤ upper), which is less than one
    # for a discrete distribution)
    return if d.lower !== nothing && x < d.lower
        one(result)
    elseif d.upper !== nothing && x > d.upper
        zero(result)
    else
        result
    end
end

function logccdf(d::Truncated, x::Real)
    d0 = d.untruncated
    result = if _upper_below_median(d)
        # P(x < X ≤ upper) / P(lower ≤ X ≤ upper)
        #   = (1 - P(X ≤ x) / P(X ≤ upper)) / (1 - P(X < lower) / P(X ≤ upper))
        logcdf_upper = _logcdf_upper(d)
        _log1mexp_ratio(logcdf(d0, x), logcdf_upper) - _log1mexp_ratio(d.loglcdf, logcdf_upper)
    else
        # P(x < X ≤ upper) / P(lower ≤ X ≤ upper)
        #   = P(X > x) (1 - P(X > upper) / P(X > x)) / P(lower ≤ X ≤ upper)
        logccdf_x = logccdf(d0, x)
        (logccdf_x - d.logtp) + _log1mexp_ratio(_logccdf_upper(d), logccdf_x)
    end
    # Special cases for values outside of the support to avoid e.g. NaN issues with `Binomial`
    # (at `x == lower` the result is P(X > lower | lower ≤ X ≤ upper), which is less than one
    # for a discrete distribution)
    return if d.lower !== nothing && x < d.lower
        zero(result)
    elseif d.upper !== nothing && x > d.upper
        oftype(result, -Inf)
    else
        result
    end
end

## random number generation

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
        # computation in log-space fixes numerical issues if d.tp is small (#1548); above the
        # median, from the upper tail: P(X > x) = P(X > upper) + U P(lower ≤ X ≤ upper) with the
        # uniform U = exp(-E)
        x = if _lower_above_median(d)
            _invlogccdf(d0, logaddexp(_logccdf_upper(d), d.logtp - randexp(rng)))
        else
            _invlogcdf(d0, logaddexp(d.loglcdf, d.logtp - randexp(rng)))
        end
        return _clamp_to_support(d, x)
    end
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
