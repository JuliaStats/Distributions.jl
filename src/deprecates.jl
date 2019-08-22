#### Deprecate on 0.6 (to be removed on 0.7)

@Base.deprecate expected_logdet meanlogdet

function probs(d::ContiguousUnivariateDistribution)
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

@deprecate pdf(d::ContiguousUnivariateDistribution) pdf.(Ref(d), support(d))

# No longer proposing use of many const aliases
const ValueSupport = Support{Float64}

const CountableUnivariateDistribution{C<:CountableSupport} =
    UnivariateDistribution{C}
const ContiguousUnivariateDistribution{S<:Integer} =
    CountableUnivariateDistribution{ContiguousSupport{S}}
const ContinuousUnivariateDistribution{T<:Number} =
    UnivariateDistribution{ContinuousSupport{T}}

const CountableMultivariateDistribution{C<:CountableSupport} =
    MultivariateDistribution{C}
const ContiguousMultivariateDistribution{S<:Integer} =
    CountableMultivariateDistribution{ContiguousSupport{S}}
const ContinuousMultivariateDistribution{T<:Number} =
    MultivariateDistribution{ContinuousSupport{T}}

const CountableMatrixDistribution{C<:CountableSupport} =
    MatrixDistribution{C}
const ContiguousMatrixDistribution{S<:Integer} =
    CountableMatrixDistribution{ContiguousSupport{S}}
const ContinuousMatrixDistribution{T<:Number} =
    MatrixDistribution{ContinuousSupport{T}}
