immutable Exponential <: ContinuousUnivariateDistribution
    scale::Float64 # note: scale not rate
    function Exponential(sc::Real)
        sc > zero(sc) || error("scale must be positive")
        new(float64(sc))
    end
end

Exponential() = Exponential(1.0)

## Support
@continuous_distr_support Exponential 0.0 Inf

## Properties
scale(d::Exponential) = d.scale
rate(d::Exponential) = 1.0 / d.scale

mean(d::Exponential) = d.scale
median(d::Exponential) = d.scale * log(2.0)
mode(d::Exponential) = 0.0

var(d::Exponential) = d.scale * d.scale
skewness(d::Exponential) = 2.0
kurtosis(d::Exponential) = 6.0

entropy(d::Exponential) = 1.0 - log(1.0 / d.scale)

## Functions
pdf(d::Exponential, x::Real) = x < zero(x) ? 0.0 : exp(-x / d.scale) / d.scale
logpdf(d::Exponential, x::Real) =  x < zero(x) ? -Inf : -x / d.scale - log(d.scale)

cdf(d::Exponential, q::Real) = q <= zero(q) ? 0.0 : -expm1(-q / d.scale)
ccdf(d::Exponential, q::Real) = q <= zero(q) ? 1.0 : exp(-q / d.scale)
logcdf(d::Exponential, q::Real) = q > zero(q) ? log1mexp(-q / d.scale) : -Inf
logccdf(d::Exponential, q::Real) = q <= zero(q) ? 0.0 : -q / d.scale

quantile(d::Exponential, p::Real) = @check_quantile p -d.scale * log1p(-p)
cquantile(d::Exponential, p::Real) = @check_quantile p -d.scale * log(p)
invlogcdf(d::Exponential, lp::Real) = @checkinvlogcdf lp -d.scale * log1mexp(lp)
invlogccdf(d::Exponential, lp::Real) = @checkinvlogcdf lp -d.scale * lp 

mgf(d::Exponential, t::Real) = 1.0/(1.0 - t * d.scale)
cf(d::Exponential, t::Real) = 1.0/(1.0 - t * im * d.scale)

gradlogpdf(d::Exponential, x::Real) = insupport(Exponential, x) ? - 1.0 / d.scale : 0.0

## Sampling
rand(d::Exponential) = d.scale * Base.Random.randmtzig_exprnd()

## Fit model
immutable ExponentialStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    sw::Float64   # sum of sample weights

    ExponentialStats(sx::Real, sw::Real) = new(float64(sx), float64(sw))
end

suffstats(::Type{Exponential}, x::Array) = ExponentialStats(sum(x), length(x))
suffstats(::Type{Exponential}, x::Array, w::Array) = ExponentialStats(dot(x, w), sum(w))

fit_mle(::Type{Exponential}, ss::ExponentialStats) = Exponential(ss.sx / ss.sw)
