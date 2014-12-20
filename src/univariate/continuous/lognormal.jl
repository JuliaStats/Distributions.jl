immutable LogNormal <: ContinuousUnivariateDistribution
    nrmd::Normal

    LogNormal(μ::Real, σ::Real) = new(Normal(μ, σ))
    LogNormal(μ::Real) = new(Normal(μ))
    LogNormal() = new(Normal())
end

@distr_support LogNormal 0.0 Inf

show(io::IO, d::LogNormal) = ((μ, σ) = params(d); show_oneline(io, d, [(:μ, μ), (:σ, σ)]))

#### Parameters

params(d::LogNormal) = params(d.nrmd)


#### Statistics

meanlogx(d::LogNormal) = mean(d.nrmd)
varlogx(d::LogNormal) = var(d.nrmd)
stdlogx(d::LogNormal) = std(d.nrmd)

mean(d::LogNormal) = ((μ, σ) = params(d); exp(μ + 0.5 * σ^2))
median(d::LogNormal) = exp(median(d.nrmd))
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

pdf(d::LogNormal, x::Float64) = pdf(d.nrmd, log(x)) / x
logpdf(d::LogNormal, x::Float64) = (lx = log(x); logpdf(d.nrmd, lx) - lx)

cdf(d::LogNormal, x::Float64) = x > 0.0 ? cdf(d.nrmd, log(x)) : 0.0
ccdf(d::LogNormal, x::Float64) = x > 0.0 ? ccdf(d.nrmd, log(x)) : 1.0
logcdf(d::LogNormal, x::Float64) = x > 0.0 ? logcdf(d.nrmd, log(x)) : -Inf
logccdf(d::LogNormal, x::Float64) = x > 0.0 ? logccdf(d.nrmd, log(x)) : 0.0

quantile(d::LogNormal, p::Float64) = exp(quantile(d.nrmd, p))
cquantile(d::LogNormal, p::Float64) = exp(cquantile(d.nrmd, p))
invlogcdf(d::LogNormal, lp::Float64) = exp(invlogcdf(d.nrmd, lp))
invlogccdf(d::LogNormal, lp::Float64) = exp(invlogccdf(d.nrmd, lp))

function gradlogpdf(d::LogNormal, x::Float64)
    (μ, σ) = params(d)
    x > 0.0 ? - ((log(x) - μ) / (σ^2) + 1.0) / x : 0.0
end

# mgf(d::LogNormal)
# cf(d::LogNormal)


#### Sampling

rand(d::LogNormal) = exp(rand(d.nrmd))


## Fitting

function fit_mle{T <: Real}(::Type{LogNormal}, x::Array{T})
    lx = log(x)
    μ, σ = mean_and_std(lx)
    LogNormal(μ, σ)
end


