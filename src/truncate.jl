"""
    truncated(d, l, u):

Truncate a distribution between `l` and `u`.
Builds the most appropriate distribution for the type of `d`,
the fallback is constructing a `Truncated` wrapper.

To implement a specialized truncated form for a distribution `D`,
the method `truncate(d::D, l::T, u::T) where {T <: Real}`
should be implemented.

# Arguments
- `d::UnivariateDistribution`: The original distribution.
- `l::Real`: The lower bound of the truncation, which can be a finite value or `-Inf`.
- `u::Real`: The upper bound of the truncation, which can be a finite value of `Inf`.

Throws an error if `l >= u`.
"""
function truncated(d::UnivariateDistribution, l::Real, u::Real)
    return truncated(d, promote(l, u)...)
end

function truncated(d::UnivariateDistribution, l::T, u::T) where {T <: Real}
    l < u || error("lower bound should be less than upper bound.")
    T2 = promote_type(T, eltype(d))
    lcdf = isinf(l) ? zero(T2) : T2(cdf(d, l))
    ucdf = isinf(u) ? one(T2) : T2(cdf(d, u))
    tp = ucdf - lcdf
    Truncated(d, promote(l, u, lcdf, ucdf, tp, log(tp))...)
end

truncated(d::UnivariateDistribution, l::Integer, u::Integer) = truncated(d, float(l), float(u))

"""
    Truncated(d, l, u):

Create a generic wrapper for a truncated distribution.
Prefer calling the function `truncated(d, l, u)`, which can choose the appropriate
representation of the truncated distribution.

# Arguments
- `d::UnivariateDistribution`: The original distribution.
- `l::Real`: The lower bound of the truncation, which can be a finite value or `-Inf`.
- `u::Real`: The upper bound of the truncation, which can be a finite value of `Inf`.
"""
struct Truncated{D<:UnivariateDistribution, S<:ValueSupport, T <: Real} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::T      # lower bound
    upper::T      # upper bound
    lcdf::T       # cdf of lower bound
    ucdf::T       # cdf of upper bound

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)
    function Truncated(d::UnivariateDistribution, l::T, u::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real}
        new{typeof(d), value_support(typeof(d)), T}(d, l, u, lcdf, ucdf, tp, logtp)
    end
end

### Constructors
function Truncated(d::UnivariateDistribution, l::T, u::T) where {T <: Real}
    l < u || error("lower bound should be less than upper bound.")
    T2 = promote_type(T, eltype(d))
    lcdf = isinf(l) ? zero(T2) : T2(cdf(d, l))
    ucdf = isinf(u) ? one(T2) : T2(cdf(d, u))
    tp = ucdf - lcdf
    Truncated(d, promote(l, u, lcdf, ucdf, tp, log(tp))...)
end

Truncated(d::UnivariateDistribution, l::Integer, u::Integer) = Truncated(d, float(l), float(u))

@deprecate Truncated(d::UnivariateDistribution, l::Real, u::Real) truncated(d, l, u)

params(d::Truncated) = tuple(params(d.untruncated)..., d.lower, d.upper)
partype(d::Truncated) = partype(d.untruncated)
Base.eltype(::Type{Truncated{D, S, T} } ) where {D, S, T} = T

### range and support

islowerbounded(d::Truncated) = islowerbounded(d.untruncated) || isfinite(d.lower)
isupperbounded(d::Truncated) = isupperbounded(d.untruncated) || isfinite(d.upper)

minimum(d::Truncated) = max(minimum(d.untruncated), d.lower)
maximum(d::Truncated) = min(maximum(d.untruncated), d.upper)

function insupport(d::Truncated{D,<:Union{Discrete,Continuous}}, x::Real) where {D<:UnivariateDistribution}
    return d.lower <= x <= d.upper && insupport(d.untruncated, x)
end

### evaluation

quantile(d::Truncated, p::Real) = quantile(d.untruncated, d.lcdf + p * d.tp)

function _pdf(d::Truncated, x::T) where {T<:Real}
    if d.lower <= x <= d.upper
        pdf(d.untruncated, x) / d.tp
    else
        zero(T)
    end
end

function pdf(d::Truncated{<:ContinuousUnivariateDistribution}, x::T) where {T<:Real}
    _pdf(d, float(x))
end

function pdf(d::Truncated{D}, x::T) where {D<:DiscreteUnivariateDistribution, T<:Real}
    isinteger(x) || return zero(float(T))
    _pdf(d, x)
end

function pdf(d::Truncated{D}, x::T) where {D<:DiscreteUnivariateDistribution, T<:Integer}
    _pdf(d, float(x))
end

function _logpdf(d::Truncated, x::T) where {T<:Real}
    if d.lower <= x <= d.upper
        logpdf(d.untruncated, x) - d.logtp
    else
        TF = float(T)
        -TF(Inf)
    end
end

function logpdf(d::Truncated{D}, x::T) where {D<:DiscreteUnivariateDistribution, T<:Real}
    TF = float(T)
    isinteger(x) || return -TF(Inf)
    return _logpdf(d, x)
end

function logpdf(d::Truncated{D}, x::Integer) where {D<:DiscreteUnivariateDistribution}
    _logpdf(d, x)
end

function logpdf(d::Truncated{D, Continuous}, x::T) where {D<:ContinuousUnivariateDistribution, T<:Real}
    _logpdf(d, x)
end

# fallback to avoid method ambiguities
_cdf(d::Truncated, x::T) where {T<:Real} =
    x <= d.lower ? zero(T) :
    x >= d.upper ? one(T) :
    (cdf(d.untruncated, x) - d.lcdf) / d.tp

cdf(d::Truncated, x::Real) = _cdf(d, x)
cdf(d::Truncated, x::Integer) = _cdf(d, float(x)) # float conversion for stability

function _logcdf(d::Truncated, x::T) where {T<:Real}
    TF = float(T)
    if x <= d.lower
        -TF(Inf)
    elseif x >= d.upper
        zero(TF)
    else
        log(cdf(d.untruncated, x) - d.lcdf) - d.logtp
    end
end

logcdf(d::Truncated, x::Real) = _logcdf(d, x)
logcdf(d::Truncated, x::Integer) = _logcdf(d, x)

_ccdf(d::Truncated, x::T) where {T<:Real} =
    x <= d.lower ? one(T) :
    x >= d.upper ? zero(T) :
    (d.ucdf - cdf(d.untruncated, x)) / d.tp

ccdf(d::Truncated, x::Real) = _ccdf(d, x)
ccdf(d::Truncated, x::Integer) = _ccdf(d, float(x))

function _logccdf(d::Truncated, x::T) where {T<:Real}
    TF = float(T)
    if x <= d.lower
        zero(TF)
    elseif x >= d.upper
        -TF(Inf)
    else
        log(d.ucdf - cdf(d.untruncated, x)) - d.logtp
    end
end

logccdf(d::Truncated, x::Real) = _logccdf(d, x)
logccdf(d::Truncated, x::Integer) = _logccdf(d, x)

## random number generation

function _rand!(rng::AbstractRNG, d::Truncated)
    d0 = d.untruncated
    if d.tp > 0.25
        while true
            r = _rand!(rng, d0)
            if d.lower <= r <= d.upper
                return r
            end
        end
    else
        return quantile(d0, d.lcdf + rand(rng) * d.tp)
    end
end


## show

function show(io::IO, d::Truncated)
    print(io, "Truncated(")
    d0 = d.untruncated
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    print(io, ", range=($(d.lower), $(d.upper)))")
    uml && println(io)
end

_use_multline_show(d::Truncated) = _use_multline_show(d.untruncated)


### specialized truncated distributions

include(joinpath("truncated", "normal.jl"))
include(joinpath("truncated", "exponential.jl"))
include(joinpath("truncated", "uniform.jl"))
