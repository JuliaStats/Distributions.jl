immutable Exponential <: ContinuousUnivariateDistribution
    β::Float64 		# note: scale not rate

    function Exponential(β::Real)
        β > zero(β) || error("scale must be positive")
        @compat new(Float64(β))
    end

    Exponential() = new(1.0)
end

@distr_support Exponential 0.0 Inf


#### Parameters

scale(d::Exponential) = d.β
rate(d::Exponential) = 1.0 / d.β

params(d::Exponential) = (d.β,)


#### Statistics

mean(d::Exponential) = scale(d)

median(d::Exponential) = logtwo * scale(d)

mode(d::Exponential) = 0.0

var(d::Exponential) = scale(d)^2

skewness(d::Exponential) = 2.0

kurtosis(d::Exponential) = 6.0

entropy(d::Exponential) = 1.0 + log(scale(d))


#### Evaluation

zval(d::Exponential, x::Float64) = x / d.β
xval(d::Exponential, z::Float64) = z * d.β

pdf(d::Exponential, x::Float64) = (λ = rate(d); x < 0.0 ? 0.0 : λ * exp(-λ * x))
logpdf(d::Exponential, x::Float64) =  (λ = rate(d); x < 0.0 ? -Inf : log(λ) - λ * x)

cdf(d::Exponential, x::Float64) = x > 0.0 ? -expm1(-zval(d, x)) : 0.0
ccdf(d::Exponential, x::Float64) = x > 0.0 ? exp(-zval(d, x)) : 0.0
logcdf(d::Exponential, x::Float64) = x > 0.0 ? log1mexp(-zval(d, x)) : -Inf
logccdf(d::Exponential, x::Float64) = x > 0.0 ? -zval(d, x) : 0.0

quantile(d::Exponential, p::Float64) = -xval(d, log1p(-p))
cquantile(d::Exponential, p::Float64) = -xval(d, log(p))
invlogcdf(d::Exponential, lp::Float64) = -xval(d, log1mexp(lp))
invlogccdf(d::Exponential, lp::Float64) = -xval(d, lp)

gradlogpdf(d::Exponential, x::Float64) = x > 0.0 ? -rate(d) : 0.0

mgf(d::Exponential, t::Real) = 1.0/(1.0 - t * scale(d))
cf(d::Exponential, t::Real) = 1.0/(1.0 - t * im * scale(d))


#### Sampling

rand(d::Exponential) = xval(d, randexp())


#### Fit model

immutable ExponentialStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    sw::Float64   # sum of sample weights

    @compat ExponentialStats(sx::Real, sw::Real) = new(Float64(sx), Float64(sw))
end

suffstats(::Type{Exponential}, x::Array) = ExponentialStats(sum(x), length(x))
suffstats(::Type{Exponential}, x::Array, w::Array) = ExponentialStats(dot(x, w), sum(w))

fit_mle(::Type{Exponential}, ss::ExponentialStats) = Exponential(ss.sx / ss.sw)
