immutable LogNormal <: ContinuousUnivariateDistribution
    meanlog::Float64
    sdlog::Float64
    function LogNormal(ml::Real, sdl::Real)
    	sdl > zero(sdl) || error("sdlog must be positive")
    	new(float64(ml), float64(sdl))
    end
end

LogNormal(ml::Real) = LogNormal(ml, 1.0)
LogNormal() = LogNormal(0.0, 1.0)

## Support
@continuous_distr_support LogNormal 0.0 Inf

## Properties
mean(d::LogNormal) = exp(d.meanlog + d.sdlog^2 / 2)

median(d::LogNormal) = exp(d.meanlog)

mode(d::LogNormal) = exp(d.meanlog - d.sdlog^2)

function var(d::LogNormal)
    sigsq = d.sdlog^2
    (exp(sigsq) - 1) * exp(2d.meanlog + sigsq)
end

function skewness(d::LogNormal)
    (exp(d.sdlog^2) + 2.0) * sqrt(exp(d.sdlog^2) - 1.0)
end

function kurtosis(d::LogNormal)
   exp(4.0 * d.sdlog^2) + 2.0 * exp(3.0 * d.sdlog^2) +
        3.0 * exp(2.0 * d.sdlog^2) - 6.0
end

entropy(d::LogNormal) = 0.5 + 0.5 * log(2.0 * pi * d.sdlog^2) + d.meanlog

## Functions
pdf(d::LogNormal, x::Real) = pdf(Normal(d.meanlog,d.sdlog),log(x))/x
logpdf(d::LogNormal, x::Real) = (lx = log(x); logpdf(Normal(d.meanlog,d.sdlog),lx)-lx)

cdf(d::LogNormal, q::Real) = q <= zero(q) ? 0.0 : cdf(Normal(d.meanlog,d.sdlog),log(q))
ccdf(d::LogNormal, q::Real) = q <= zero(q) ? 1.0 : ccdf(Normal(d.meanlog,d.sdlog),log(q))
logcdf(d::LogNormal, q::Real) = q <= zero(q) ? -Inf : logcdf(Normal(d.meanlog,d.sdlog),log(q))
logccdf(d::LogNormal, q::Real) = q <= zero(q) ? 0.0 : logccdf(Normal(d.meanlog,d.sdlog),log(q))

quantile(d::LogNormal, p::Real) = exp(quantile(Normal(d.meanlog,d.sdlog),p))
cquantile(d::LogNormal, p::Real) = exp(cquantile(Normal(d.meanlog,d.sdlog),p))
invlogcdf(d::LogNormal, p::Real) = exp(invlogcdf(Normal(d.meanlog,d.sdlog),p))
invlogccdf(d::LogNormal, p::Real) = exp(invlogccdf(Normal(d.meanlog,d.sdlog),p))

function gradlogpdf(d::LogNormal, x::Real)
  insupport(LogNormal, x) ? - ((log(x) - d.meanlog) / (d.sdlog^2) + 1.0) / x : 0.0
end

# mgf(d::LogNormal)
# cf(d::LogNormal)


## Sampling
rand(d::LogNormal) = exp(rand(Normal(d.meanlog,d.sdlog)))

## Fitting
function fit_mle{T <: Real}(::Type{LogNormal}, x::Array{T})
    lx = log(x)
    LogNormal(mean(lx), std(lx))
end
