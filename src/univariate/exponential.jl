immutable Exponential <: ContinuousUnivariateDistribution
    scale::Float64 # note: scale not rate
    function Exponential(sc::Real)
        sc > zero(sc) || error("scale must be positive")
        new(float64(sc))
    end
end

Exponential() = Exponential(1.0)

@continuous_distr_support Exponential 0.0 Inf

scale(d::Exponential) = d.scale
rate(d::Exponential) = 1.0 / d.scale

cdf(d::Exponential, q::Real) = q <= zero(q) ? 0.0 : -expm1(-q / d.scale)

logcdf(d::Exponential, q::Real) = q > zero(q) ? log1mexp(-q / d.scale) : -Inf

ccdf(d::Exponential, q::Real) = q <= zero(q) ? 1.0 : exp(-q / d.scale)

logccdf(d::Exponential, q::Real) = q <= zero(q) ? 0.0 : -q / d.scale

invlogcdf(d::Exponential, lp::Real) = lp > zero(lp) ? NaN : -d.scale * log1mexp(lp)

invlogccdf(d::Exponential, lp::Real) = lp <= zero(lp) ? -d.scale * lp : NaN

entropy(d::Exponential) = 1.0 - log(1.0 / d.scale)

kurtosis(d::Exponential) = 6.0

mean(d::Exponential) = d.scale

median(d::Exponential) = d.scale * log(2.0)

mgf(d::Exponential, t::Real) = 1.0/(1.0 - t * d.scale)

cf(d::Exponential, t::Real) = (1.0 - t * im * d.scale)^(-1)

mode(d::Exponential) = 0.0
modes(d::Exponential) = [0.0]

pdf(d::Exponential, x::Real) = x < zero(x) ? 0.0 : exp(-x / d.scale) / d.scale

logpdf(d::Exponential, x::Real) =  x < zero(x) ? -Inf : -x / d.scale - log(d.scale)

quantile(d::Exponential, p::Real) = zero(p) <= p <= one(p) ? -d.scale * log1p(-p) : NaN

cquantile(d::Exponential, p::Real) = zero(p) <= p <= one(p) ? -d.scale * log(p) : NaN

rand(d::Exponential) = d.scale * Base.Random.randmtzig_exprnd()

function rand!(d::Exponential, A::Array{Float64})
    Base.Random.randmtzig_fill_exprnd!(A)
    for i in 1:length(A)
        A[i] *= d.scale
    end
    A
end

skewness(d::Exponential) = 2.0

var(d::Exponential) = d.scale * d.scale

function gradloglik(d::Exponential, x::Float64)
  insupport(Exponential, x) ? - 1.0 / d.scale : 0.0
end

### handling support

isupperbounded(::Union(Exponential, Type{Exponential})) = false
islowerbounded(::Union(Exponential, Type{Exponential})) = true
isbounded(::Union(Exponential, Type{Exponential})) = false

hasfinitesupport(::Union(Exponential, Type{Exponential})) = false
minimum(::Union(Exponential, Type{Exponential})) = 0.0
maximum(::Union(Exponential, Type{Exponential})) = Inf

insupport(::Union(Exponential, Type{Exponential}), x::Real) = x >= 0.0


## Fit model

immutable ExponentialStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    sw::Float64   # sum of sample weights

    ExponentialStats(sx::Real, sw::Real) = new(float64(sx), float64(sw))
end

suffstats(::Type{Exponential}, x::Array) = ExponentialStats(sum(x), length(x))
    
suffstats(::Type{Exponential}, x::Array, w::Array) = ExponentialStats(dot(x, w), sum(w))

fit_mle(::Type{Exponential}, ss::ExponentialStats) = Exponential(ss.sx / ss.sw)
