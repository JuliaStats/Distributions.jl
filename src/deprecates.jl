#### Deprecate on 0.6 (to be removed on 0.7)

@Base.deprecate expected_logdet meanlogdet

function probs(d::DiscreteUnivariateDistribution)
    Base.depwarn("probs(d::$(typeof(d))) is deprecated. Please use pdf(d) instead.", :probs)
    return probs(d)
end

function Binomial(n::Real, p::Real)
    Base.depwarn("Binomial(n::Real, p) is deprecated. Please use Binomial(n::Integer, p) instead.", :Binomial)
    Binomial(Int(n), p)
end

function Binomial(n::Real)
    Base.depwarn("Binomial(n::Real) is deprecated. Please use Binomial(n::Integer) instead.", :Binomial)
    Binomial(Int(n))
end

function BetaBinomial(n::Real, α::Real, β::Real)
    Base.depwarn("BetaBinomial(n::Real, α, β) is deprecated. Please use BetaBinomial(n::Integer, α, β) instead.", :BetaBinomial)
    BetaBinomial(Int(n), α, β)
end

Base.@deprecate Arcsine(b::Real) Arcsine(b=b)
Base.@deprecate Beta(α::Real) Beta(α=α, β=α)
Base.@deprecate BetaPrime(α::Real) BetaPrime(α=α, β=α)
Base.@deprecate Biweight(μ::Real) Biweight(μ=μ)
Base.@deprecate Cauchy(μ::Real) Cauchy(μ=μ)
Base.@deprecate Cosine(μ::Real) Cosine(μ=μ)
Base.@deprecate Epanechnikov(μ::Real) Epanechnikov(μ=μ)
Base.@deprecate Erlang(α::Int) Erlang(α=α)
Base.@deprecate Gamma(α::Real) Gamma(α=α)
Base.@deprecate Gumbel(μ::Real) Gumbel(μ=μ)
Base.@deprecate Frechet(α::Real) Frechet(α=α)
Base.@deprecate GeneralizedPareto(σ::Real, ξ::Real) GeneralizedPareto(σ=σ, ξ=ξ)
Base.@deprecate InverseGamma(α::Real) InverseGamma(α=α)
Base.@deprecate InverseGaussian(μ::Real) InverseGaussian(μ=μ)
Base.@deprecate Laplace(μ::Real) Laplace(μ=μ)
Base.@deprecate Levy(μ::Real) Levy(μ=μ)
Base.@deprecate Logistic(μ::Real) Logistic(μ=μ)
Base.@deprecate LogNormal(μ::Real) LogNormal(μ=μ)
Base.@deprecate Normal(μ::Real) Normal(μ=μ)


# vectorized versions
for fun in [:pdf, :logpdf,
            :cdf, :logcdf,
            :ccdf, :logccdf,
            :invlogcdf, :invlogccdf,
            :quantile, :cquantile]

    _fun! = Symbol('_', fun, '!')
    fun! = Symbol(fun, '!')

    @eval begin
        @deprecate ($_fun!)(r::AbstractArray, d::UnivariateDistribution, X::AbstractArray) r .= ($fun).(d, X) false
        @deprecate ($fun!)(r::AbstractArray, d::UnivariateDistribution, X::AbstractArray) r .= ($fun).(d, X) false
        @deprecate ($fun)(d::UnivariateDistribution, X::AbstractArray) ($fun).(d, X)
    end
end

@deprecate pdf(d::DiscreteUnivariateDistribution) pdf.(Ref(d), support(d))
