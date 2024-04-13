#### Deprecate on 0.6 (to be removed on 0.7)

@Base.deprecate expected_logdet meanlogdet

function probs(d::DiscreteUnivariateDistribution)
    Base.depwarn("probs(d::$(typeof(d))) is deprecated. Please use pdf(d) instead.", :probs)
    return pdf(d)
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
        @deprecate ($_fun!)(r::AbstractArray{<:Real}, d::UnivariateDistribution, X::AbstractArray{<:Real}) r .= Base.Fix1($fun, d).(X) false
        @deprecate ($fun!)(r::AbstractArray{<:Real}, d::UnivariateDistribution, X::AbstractArray{<:Real}) r .= Base.Fix1($fun, d).(X) false
        @deprecate ($fun)(d::UnivariateDistribution, X::AbstractArray{<:Real}) map(Base.Fix1($fun, d), X)
    end
end

@deprecate pdf(d::DiscreteUnivariateDistribution) map(Base.Fix1(pdf, d), support(d))

# Wishart constructors
@deprecate Wishart(df::Real, S::AbstractPDMat, warn::Bool) Wishart(df, S)
@deprecate Wishart(df::Real, S::Matrix, warn::Bool) Wishart(df, S)
@deprecate Wishart(df::Real, S::Cholesky, warn::Bool) Wishart(df, S)

# Deprecate 3 arguments expectation and once with function in second place
@deprecate expectation(distr::DiscreteUnivariateDistribution, g::Function, epsilon::Real) expectation(g, distr; epsilon=epsilon) false
@deprecate expectation(distr::ContinuousUnivariateDistribution, g::Function, epsilon::Real) expectation(g, distr) false
@deprecate expectation(distr::Union{UnivariateDistribution,MultivariateDistribution}, g::Function; kwargs...) expectation(g, distr; kwargs...) false

# Deprecate `MatrixReshaped`
const MatrixReshaped{S<:ValueSupport,D<:MultivariateDistribution{S}} = ReshapedDistribution{2,S,D}
@deprecate MatrixReshaped(
    d::MultivariateDistribution, n::Integer, p::Integer=n
) reshape(d, (n, p))

for D in (:InverseWishart, :LKJ, :MatrixBeta, :MatrixFDist, :Wishart)
    @eval @deprecate dim(d::$D) size(d, 1)
end
