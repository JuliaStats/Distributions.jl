immutable LogNormal <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    function LogNormal(μ::Real, σ::Real)
        σ > zero(σ) || error("σ must be positive")
        @compat new(Float64(μ), Float64(σ))
    end

    LogNormal(μ::Real) = @compat new(Float64(μ), 1.0)
    LogNormal() = new(0.0, 1.0)
end

@distr_support LogNormal 0.0 Inf

#### Parameters

params(d::LogNormal) = (d.μ, d.σ)

#### Statistics

meanlogx(d::LogNormal) = d.μ
varlogx(d::LogNormal) = abs2(d.σ)
stdlogx(d::LogNormal) = d.σ

mean(d::LogNormal) = ((μ, σ) = params(d); exp(μ + 0.5 * σ^2))
median(d::LogNormal) = exp(d.μ)
mode(d::LogNormal) = ((μ, σ) = params(d); exp(μ - σ^2))

function var(d::LogNormal)
    (μ, σ) = params(d)
    σ2 = σ^2
    (exp(σ2) - 1.0) * exp(2.0 * μ + σ2)
end

function skewness(d::LogNormal)
    σ2 = varlogx(d)
    e = exp(σ2)
    (e + 2.0) * sqrt(e - 1.0)
end

function kurtosis(d::LogNormal)
    σ2 = varlogx(d)
    e = exp(σ2)
    e2 = e * e
    e3 = e2 * e
    e4 = e3 * e
    e4 + 2.0 * e3 + 3.0 * e2 - 6.0
end

function entropy(d::LogNormal)
    (μ, σ) = params(d)
    0.5 * (1.0 + log(twoπ * σ^2)) + μ
end


#### Evalution

pdf(d::LogNormal, x::Float64) = normpdf(d.μ, d.σ, log(x)) / x
logpdf(d::LogNormal, x::Float64) = (lx = log(x); normlogpdf(d.μ, d.σ, lx) - lx)

cdf(d::LogNormal, x::Float64) = x > 0.0 ? normcdf(d.μ, d.σ, log(x)) : 0.0
ccdf(d::LogNormal, x::Float64) = x > 0.0 ? normccdf(d.μ, d.σ, log(x)) : 1.0
logcdf(d::LogNormal, x::Float64) = x > 0.0 ? normlogcdf(d.μ, d.σ, log(x)) : -Inf
logccdf(d::LogNormal, x::Float64) = x > 0.0 ? normlogccdf(d.μ, d.σ, log(x)) : 0.0

quantile(d::LogNormal, q::Float64) = exp(norminvcdf(d.μ, d.σ, q))
cquantile(d::LogNormal, q::Float64) = exp(norminvccdf(d.μ, d.σ, q))
invlogcdf(d::LogNormal, lq::Float64) = exp(norminvlogcdf(d.μ, d.σ, lq))
invlogccdf(d::LogNormal, lq::Float64) = exp(norminvlogccdf(d.μ, d.σ, lq))

function gradlogpdf(d::LogNormal, x::Float64)
    (μ, σ) = params(d)
    x > 0.0 ? - ((log(x) - μ) / (σ^2) + 1.0) / x : 0.0
end

# mgf(d::LogNormal)
# cf(d::LogNormal)


#### Sampling

rand(d::LogNormal) = exp(randn() * d.σ + d.μ)

## Fitting

function fit_mle{T <: Real}(::Type{LogNormal}, x::AbstractArray{T})
    lx = log(x)
    μ, σ = mean_and_std(lx)
    LogNormal(μ, σ)
end
