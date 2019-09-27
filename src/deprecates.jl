#### Deprecate on 0.6 (to be removed on 0.7)

@Base.deprecate expected_logdet meanlogdet

function probs(d::UnivariateDistribution{<:ContiguousSupport})
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

@deprecate pdf(d::UnivariateDistribution{<:ContiguousSupport}) pdf.(Ref(d), support(d))

# No longer proposing use of many const aliases
Base.@deprecate_binding ValueSupport Support{Float64}
Base.@deprecate_binding Discrete     ContiguousSupport{Int}
Base.@deprecate_binding Continuous   ContinuousSupport{Float64}

const DiscreteDistribution{F<:VariateForm}   = Distribution{F,Discrete}
const ContinuousDistribution{F<:VariateForm} = Distribution{F,Continuous}

Base.@deprecate_binding DiscreteUnivariateDistribution     Distribution{Univariate,    Discrete}
Base.@deprecate_binding ContinuousUnivariateDistribution   Distribution{Univariate,    Continuous}
Base.@deprecate_binding DiscreteMultivariateDistribution   Distribution{Multivariate,  Discrete}
Base.@deprecate_binding ContinuousMultivariateDistribution Distribution{Multivariate,  Continuous}
Base.@deprecate_binding DiscreteMatrixDistribution         Distribution{Matrixvariate, Discrete}
Base.@deprecate_binding ContinuousMatrixDistribution       Distribution{Matrixvariate, Continuous}
