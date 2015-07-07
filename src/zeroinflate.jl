
immutable ZeroInflated{D<:UnivariateDistribution} <: ContinuousUnivariateDistribution
    uninflated::D       # the original distribution (uninflated)
    zeroProb::Float64   # extra weight at zero
    # upper::Float64      # upper bound
    # lcdf::Float64       # cdf of lower bound
    # ucdf::Float64       # cdf of upper bound

    # tp::Float64         # the probability of the trancated part, i.e. ucdf - lcdf
    # logtp::Float64      # log(tp), i.e. log(ucdf - lcdf)
end

### Constructors

function ZeroInflated(d::ContinuousUnivariateDistribution, p::Float64)
    (p >= 0.0 && p <= 1.0) || error("zero inflation probability must be within [0,1].")
    ZeroInflated{typeof(d)}(d, p)
end

@compat ZeroInflated(d::ContinuousUnivariateDistribution, p::Real) = ZeroInflated(d, Float64(p))


### range and support

islowerbounded(d::ZeroInflated) = islowerbounded(d.uninflated)
isupperbounded(d::ZeroInflated) = isupperbounded(d.uninflated)

minimum(d::ZeroInflated) = min(minimum(d.uninflated), 0.0)
maximum(d::ZeroInflated) = max(maximum(d.uninflated), 0.0)

insupport(d::ZeroInflated, x::Real) = x == 0.0 || insupport(d.uninflated, x)


### evaluation

pdf(d::ZeroInflated, x::Float64) = (1-d.zeroProb) * pdf(d.uninflated, x) + (x == 0.0 ? d.zeroProb : 0.0)

logpdf(d::ZeroInflated, x::Float64) = x == 0.0 ? log(pdf(d, x)) : logpdf(d.uninflated, x) + log(1-d.zeroProb)

cdf(d::ZeroInflated, x::Float64) = (1-d.zeroProb) * cdf(d.uninflated, x) + (x >= 0.0 ? d.zeroProb : 0.0)

logcdf(d::ZeroInflated, x::Float64) = x >= 0 ? log(cdf(d, x)) : logcdf(d.uninflated, x) + log(1-d.zeroProb)

ccdf(d::ZeroInflated, x::Float64) = (1-d.zeroProb) * ccdf(d.uninflated, x) + (x <= 0.0 ? d.zeroProb : 0.0)

logccdf(d::ZeroInflated, x::Float64) = x <= 0 ? log(ccdf(d, x)) : logccdf(d.uninflated, x) + log(1-d.zeroProb)

function quantile(d::ZeroInflated, p::Float64)
    if p < cdf(d.uninflated, 0.0)*(1-d.zeroProb)
        return quantile(d.uninflated, p/(1-d.zeroProb))
    elseif p <= cdf(d, 0.0)
        return 0.0
    end

    quantile(d.uninflated, (p - d.zeroProb)/(1-d.zeroProb))
end


## random number generation

function rand(d::ZeroInflated)
    if rand() < d.zeroProb
        return 0.0
    end

    rand(d.uninflated)
end


## show

function show(io::IO, d::ZeroInflated)
    print(io, "ZeroInflated(")
    d0 = d.uninflated
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    print(io, ", extra_zero_prob=($(d.zeroProb)))")
    uml && println(io)
end

_use_multline_show(d::ZeroInflated) = _use_multline_show(d.uninflated)




