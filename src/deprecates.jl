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
        @deprecate ($_fun!)(r::AbstractArray, d::UnivariateDistribution, X::AbstractArray) r .= ($fun).(Ref(d), X) false
        @deprecate ($fun!)(r::AbstractArray, d::UnivariateDistribution, X::AbstractArray) r .= ($fun).(Ref(d), X) false
        @deprecate ($fun)(d::UnivariateDistribution, X::AbstractArray) ($fun).(Ref(d), X)
    end
end

@deprecate pdf(d::DiscreteUnivariateDistribution) pdf.(Ref(d), support(d))

# multivariate distributions
for fun in (:pdf, :logpdf)
    _fun! = Symbol('_', fun, '!')
    fun! = Symbol(fun, '!')

    # `eachcol` requires Julia 1.1
    @static if VERSION < v"1.1"
        @eval begin
            @deprecate ($_fun!)(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix) r .= ($fun).(Ref(d), (view(x, :, i) for i in axes(x, 2))) false
            @deprecate ($fun!)(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix) r .= ($fun).(Ref(d), (view(x, :, i) for i in axes(x, 2)))
            @deprecate ($fun)(d::MultivariateDistribution, X::AbstractMatrix) ($fun).(Ref(d), (view(x, :, i) for i in axes(x, 2)))
        end
    else
        @eval begin
            @deprecate ($_fun!)(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix) r .= ($fun).(Ref(d), eachcol(X)) false
            @deprecate ($fun!)(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix) r .= ($fun).(Ref(d), eachcol(X))
            @deprecate ($fun)(d::MultivariateDistribution, X::AbstractMatrix) ($fun).(Ref(d), eachcol(X))
        end
    end
end

# matrix distributions
for fun in (:pdf, :logpdf)
    _fun! = Symbol('_', fun, '!')
    fun! = Symbol(fun, '!')

    @eval begin
        @deprecate ($_fun!)(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{<:AbstractMatrix}) r .= ($fun).(Ref(d), X) false
        @deprecate ($_fun!)(r::AbstractArray, d::MatrixDistribution, X::AbstractMatrix{<:AbstractMatrix}) r .= ($fun).(Ref(d), X) false
        @deprecate ($fun!)(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{<:AbstractMatrix}) r .= ($fun).(Ref(d), X)
        @deprecate ($fun!)(r::AbstractArray, d::MatrixDistribution, X::AbstractMatrix{<:AbstractMatrix}) r .= ($fun).(Ref(d), X)
        @deprecate ($fun)(d::MatrixDistribution, X::AbstractArray{<:AbstractMatrix}) ($fun).(Ref(d), X)
        @deprecate ($fun)(d::MatrixDistribution, X::AbstractMatrix{<:AbstractMatrix}) ($fun).(Ref(d), X)
    end
end
