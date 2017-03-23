immutable Truncated{D<:UnivariateDistribution, S<:ValueSupport} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::Real      # lower bound
    upper::Real      # upper bound
    lcdf::Real       # cdf of lower bound
    ucdf::Real       # cdf of upper bound

    tp::Real         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::Real      # log(tp), i.e. log(ucdf - lcdf)
end

### Constructors

function Truncated(d::UnivariateDistribution, l::Real, u::Real)
    l < u || error("lower bound should be less than upper bound.")
    lcdf = isinf(l) ? 0.0 : cdf(d, l)
    ucdf = isinf(u) ? 1.0 : cdf(d, u)
    tp = ucdf - lcdf
    Truncated{typeof(d),value_support(typeof(d))}(d, l, u, lcdf, ucdf, tp, log(tp))
end

Truncated(d::UnivariateDistribution, l::Real, u::Real) = Truncated(d, Real(l), Real(u))

params(d::Truncated) = tuple(params(d.untruncated)..., d.lower, d.upper)
partype(d::Truncated) = partype(d.untruncated)
### range and support

islowerbounded(d::Truncated) = islowerbounded(d.untruncated) || isfinite(d.lower)
isupperbounded(d::Truncated) = isupperbounded(d.untruncated) || isfinite(d.upper)

minimum(d::Truncated) = max(minimum(d.untruncated), d.lower)
maximum(d::Truncated) = min(maximum(d.untruncated), d.upper)

insupport{D<:UnivariateDistribution}(d::Truncated{D,Union{Discrete,Continuous}}, x::Real) =
    d.lower <= x <= d.upper && insupport(d.untruncated, x)


### evaluation

pdf(d::Truncated, x::Real) = d.lower <= x <= d.upper ? pdf(d.untruncated, x) / d.tp : 0.0

logpdf(d::Truncated, x::Real) = d.lower <= x <= d.upper ? logpdf(d.untruncated, x) - d.logtp : -Inf

cdf(d::Truncated, x::Real) = x <= d.lower ? 0.0 :
                             x >= d.upper ? 1.0 :
                             (cdf(d.untruncated, x) - d.lcdf) / d.tp

logcdf(d::Truncated, x::Real) = x <= d.lower ? -Inf :
                                x >= d.upper ? 0.0 :
                                log(cdf(d.untruncated, x) - d.lcdf) - d.logtp

ccdf(d::Truncated, x::Real) = x <= d.lower ? 1.0 :
                              x >= d.upper ? 0.0 :
                              (d.ucdf - cdf(d.untruncated, x)) / d.tp

logccdf(d::Truncated, x::Real) = x <= d.lower ? 0.0 :
                                 x >= d.upper ? -Inf :
                                 log(d.ucdf - cdf(d.untruncated, x)) - d.logtp


quantile(d::Truncated, p::Real) = quantile(d.untruncated, d.lcdf + p * d.tp)

pdf(d::Truncated, x::Int) = d.lower <= x <= d.upper ? pdf(d.untruncated, x) / d.tp : 0.0

logpdf(d::Truncated, x::Int) = d.lower <= x <= d.upper ? logpdf(d.untruncated, x) - d.logtp : -Inf

cdf(d::Truncated, x::Int) = x <= d.lower ? 0.0 :
                             x >= d.upper ? 1.0 :
                             (cdf(d.untruncated, x) - d.lcdf) / d.tp

logcdf(d::Truncated, x::Int) = x <= d.lower ? -Inf :
                                x >= d.upper ? 0.0 :
                                log(cdf(d.untruncated, x) - d.lcdf) - d.logtp

ccdf(d::Truncated, x::Int) = x <= d.lower ? 1.0 :
                              x >= d.upper ? 0.0 :
                              (d.ucdf - cdf(d.untruncated, x)) / d.tp

logccdf(d::Truncated, x::Int) = x <= d.lower ? 0.0 :
                                 x >= d.upper ? -Inf :
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
