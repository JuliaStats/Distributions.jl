immutable Erlang <: ContinuousUnivariateDistribution
    gammad::Gamma

    function Erlang(α::Real, θ::Real)
        isinteger(α) || error("Erlang shape parameter must be an integer")
        new(Gamma(α, θ))
    end

    Erlang(α::Real) = Erlang(α, 1.0)
    Erlang() = new(Gamma())
end

@distr_support Erlang 0.0 Inf

show(io::IO, d::Erlang) = ((α, β) = params(d); show_oneline(io, d, [(:α, α), (:β, β)]))

#### Parameters

shape(d::Erlang) = round(Int, shape(d.gammad))
scale(d::Erlang) = scale(d.gammad)
rate(d::Erlang) = rate(d.gammad)
params(d::Erlang) = ((α, β) = params(d.gammad); (round(Int, α), β))


#### Statistics

mean(d::Erlang) = mean(d.gammad)
var(d::Erlang) = var(d.gammad)
median(d::Erlang) = median(d.gammad)
skewness(d::Erlang) = skewness(d.gammad)
kurtosis(d::Erlang) = kurtosis(d.gammad)

mode(d::Erlang) = mode(d.gammad)
entropy(d::Erlang) = entropy(d.gammad)


#### Evaluation

pdf(d::Erlang, x::Float64) = pdf(d.gammad, x)
logpdf(d::Erlang, x::Float64) = logpdf(d.gammad, x)

cdf(d::Erlang, x::Float64) = cdf(d.gammad, x)
ccdf(d::Erlang, x::Float64) = ccdf(d.gammad, x)
logcdf(d::Erlang, x::Float64) = logcdf(d.gammad, x)
logccdf(d::Erlang, x::Float64) = logccdf(d.gammad, x)

quantile(d::Erlang, p::Float64) = quantile(d.gammad, p)
cquantile(d::Erlang, p::Float64) = cquantile(d.gammad, p)
invlogcdf(d::Erlang, lp::Float64) = invlogcdf(d.gammad, lp)
invlogccdf(d::Erlang, lp::Float64) = invlogccdf(d.gammad, lp)

mgf(d::Erlang, t::Real) = mgf(d.gammad, t)
cf(d::Erlang, t::Real) = cf(d.gammad, t)


#### Sampling

rand(d::Erlang) = rand(d.gammad)

