"""
    Truncated(d, l, u):

Construct a truncated distribution.

# Arguments
- `d::UnivariateDistribution`: The original distribution.
- `l::Real`: The lower bound of the truncation, which can be a finite value or `-Inf`.
- `u::Real`: The upper bound of the truncation, which can be a finite value of `Inf`.
"""
struct Truncated{D<:UnivariateDistribution, S<:ValueSupport} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::Float64      # lower bound
    upper::Float64      # upper bound
    lcdf::Float64       # cdf of lower bound
    ucdf::Float64       # cdf of upper bound

    tp::Float64         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::Float64      # log(tp), i.e. log(ucdf - lcdf)
end

### Constructors

function Truncated(d::UnivariateDistribution, l::Float64, u::Float64)
    l < u || error("lower bound should be less than upper bound.")
    lcdf = isinf(l) ? 0.0 : cdf(d, l)
    ucdf = isinf(u) ? 1.0 : cdf(d, u)
    tp = ucdf - lcdf
    Truncated{typeof(d),value_support(typeof(d))}(d, l, u, lcdf, ucdf, tp, log(tp))
end

Truncated(d::UnivariateDistribution, l::Real, u::Real) = Truncated(d, Float64(l), Float64(u))

params(d::Truncated) = tuple(params(d.untruncated)..., d.lower, d.upper)
partype(d::Truncated) = partype(d.untruncated)
### range and support

islowerbounded(d::Truncated) = islowerbounded(d.untruncated) || isfinite(d.lower)
isupperbounded(d::Truncated) = isupperbounded(d.untruncated) || isfinite(d.upper)

minimum(d::Truncated) = max(minimum(d.untruncated), d.lower)
maximum(d::Truncated) = min(maximum(d.untruncated), d.upper)

insupport(d::Truncated{D,Union{Discrete,Continuous}}, x::Real) where {D<:UnivariateDistribution} =
    d.lower <= x <= d.upper && insupport(d.untruncated, x)


### evaluation

quantile(d::Truncated, p::Real) = quantile(d.untruncated, d.lcdf + p * d.tp)

for f in [:pdf, :logpdf, :cdf, :logcdf, :ccdf, :logccdf]
    @eval ($f)(d::Truncated, x::Int) = ($f)(d, float(x))
end

pdf(d::Truncated, x::T) where {T<:Real} = d.lower <= x <= d.upper ? pdf(d.untruncated, x) / d.tp : zero(T)

logpdf(d::Truncated, x::T) where {T<:Real} = d.lower <= x <= d.upper ? logpdf(d.untruncated, x) - d.logtp : -T(Inf)

cdf(d::Truncated, x::T) where {T<:Real} =
    x <= d.lower ? zero(T) :
    x >= d.upper ? one(T) :
    (cdf(d.untruncated, x) - d.lcdf) / d.tp

logcdf(d::Truncated, x::T) where {T<:Real} =
    x <= d.lower ? -T(Inf) :
    x >= d.upper ? zero(T) :
    log(cdf(d.untruncated, x) - d.lcdf) - d.logtp

ccdf(d::Truncated, x::T) where {T<:Real} =
    x <= d.lower ? one(T) :
    x >= d.upper ? zero(T) :
    (d.ucdf - cdf(d.untruncated, x)) / d.tp

logccdf(d::Truncated, x::T) where {T<:Real} =
    x <= d.lower ? zero(T) :
    x >= d.upper ? -T(Inf) :
    log(d.ucdf - cdf(d.untruncated, x)) - d.logtp

## random number generation

function rand(d::Truncated)
    d0 = d.untruncated
    if d.tp > 0.25
        while true
            r = rand(d0)
            if d.lower <= r <= d.upper
                return r
            end
        end
    else
        return quantile(d0, d.lcdf + rand() * d.tp)
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
