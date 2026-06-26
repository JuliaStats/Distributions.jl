#### Deprecate on 0.6 (to be removed on 0.7)

export expected_logdet
@noinline function expected_logdet(d::Wishart)
    Base.depwarn("`expected_logdet(d::Wishart)` is deprecated, use `meanlogdet(d)` instead.", :expected_logdet; force = true)
    return meanlogdet(d)
end

@noinline function probs(d::DiscreteUnivariateDistribution)
    Base.depwarn("`probs(d::DiscreteUnivariateDistribution)` is deprecated, use `pdf(d)` instead.", :probs; force = true)
    return pdf(d)
end

@noinline function Binomial(n::Real, p::Real)
    Base.depwarn("`Binomial(n::Real, p::Real)` is deprecated, use `Binomial(Int(n), p)` instead.", :Binomial; force = true)
    Binomial(Int(n), p)
end

@noinline function Binomial(n::Real)
    Base.depwarn("`Binomial(n::Real)` is deprecated, use `Binomial(Int(n))` instead.", :Binomial; force = true)
    Binomial(Int(n))
end

@noinline function BetaBinomial(n::Real, α::Real, β::Real)
    Base.depwarn("`BetaBinomial(n::Real, α::Real, β::Real)` is deprecated, use `BetaBinomial(Int(n), α, β)` instead.", :BetaBinomial; force = true)
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
        export $(fun!)
        @noinline function ($_fun!)(r::AbstractArray{<:Real}, d::UnivariateDistribution, X::AbstractArray{<:Real})
            Base.depwarn("`$(string(_fun!))(r::AbstractArray{<:Real}, d::UnivariateDistribution, X::AbstractArray{<:Real})` is deprecated, use `r .= $(string(fun)).(d, X)` instead.", $(QuoteNode(_fun!)); force = true)
            r .= $(fun).(d, X)
        end
        @noinline function ($fun!)(r::AbstractArray{<:Real}, d::UnivariateDistribution, X::AbstractArray{<:Real})
            Base.depwarn("`$(string(fun!))(r::AbstractArray{<:Real}, d::UnivariateDistribution, X::AbstractArray{<:Real})` is deprecated, use `r .= $(string(fun)).(d, X)` instead.", $(QuoteNode(fun!)); force = true)
            r .= $(fun).(d, X)
        end
        @noinline function ($fun)(d::UnivariateDistribution, X::AbstractArray{<:Real})
            Base.depwarn("`$(string(fun))(d::UnivariateDistribution, X::AbstractArray{<:Real})` is deprecated, use `$(string(fun)).(d, X)` instead.", $(QuoteNode(fun)); force = true)
            $(fun).(d, X)
        end
    end
end

@noinline function pdf(d::DiscreteUnivariateDistribution)
    Base.depwarn("`pdf(d::DiscreteUnivariateDistribution)` is deprecated, please use `pdf.(d, support(d))` instead.", :pdf; force = true)
    pdf.(d, support(d))
end

# Wishart constructors
@noinline function Wishart(df::Real, S::AbstractPDMat, ::Bool)
    Base.depwarn("`Wishart(df::Real, S::AbstractPDMat, warn::Bool)` is deprecated, use `Wishart(df, S)` instead.", :Wishart; force = true)
    Wishart(df, S)
end
@noinline function Wishart(df::Real, S::Matrix, ::Bool)
    Base.depwarn("`Wishart(df::Real, S::Matrix, warn::Bool)` is deprecated, use `Wishart(df, S)` instead.", :Wishart; force = true)
    Wishart(df, S)
end
@noinline function Wishart(df::Real, S::Cholesky, ::Bool)
    Base.depwarn("`Wishart(df::Real, S::Cholesky, warn::Bool)` is deprecated, use `Wishart(df, S)` instead.", :Wishart; force = true)
    Wishart(df, S)
end

# Deprecate 3 arguments expectation and once with function in second place
@noinline function expectation(distr::DiscreteUnivariateDistribution, g::Function, epsilon::Real)
    Base.depwarn("`expectation(d::DiscreteUnivariateDistribution, g::Function, epsilon::Real)` is deprecated, use `expectation(g, d; epsilon)` instead.", :expectation; force = true)
    expectation(g, distr; epsilon=epsilon)
end
@noinline function expectation(distr::ContinuousUnivariateDistribution, g::Function, ::Real)
    Base.depwarn("`expectation(d::ContinuousUnivariateDistribution, g::Function, epsilon::Real)` is deprecated, use `expectation(g, d)` instead.", :expectation; force = true)
    expectation(g, distr)
end
@noinline function expectation(distr::Union{UnivariateDistribution,MultivariateDistribution}, g::Function; kwargs...)
    Base.depwarn("`expectation(d::ContinuousUnivariateDistribution, g::Function; kwargs...)` is deprecated, use `expectation(g, d; kwargs...)` instead.", :expectation; force = true)
    expectation(g, distr; kwargs...)
end

# Deprecate `MatrixReshaped`
# This is very similar to `Base.@deprecate_binding MatrixReshaped{...} ReshapedDistribution{...}`
# However, `Base.@deprecate_binding` does not support type parameters
export MatrixReshaped
const MatrixReshaped{S<:ValueSupport,D<:MultivariateDistribution{S}} = ReshapedDistribution{2,S,D}
Base.deprecate(@__MODULE__, :MatrixReshaped)
# This is very similar to `Base.@deprecate MatrixReshaped(...) reshape(...)`
# We use another (unexported!) alias here to not throw a deprecation warning/error
# Unexported aliases do not affect the type printing
# In Julia >= 1.6, instead of a new alias we could have defined a method for (ReshapedDistribution{2,S,D} where {S<:ValueSupport,D<:MultivariateDistribution{S}})
const _MatrixReshaped{S<:ValueSupport,D<:MultivariateDistribution{S}} = ReshapedDistribution{2,S,D}
function _MatrixReshaped(d::MultivariateDistribution, n::Integer, p::Integer=n)
    Base.depwarn("`MatrixReshaped(d::MultivariateDistribution, n::Integer, p::Integer=n)` is deprecated, use `reshape(d, (n, p))` instead.", :MatrixReshaped; force = true)
    return reshape(d, (n, p))
end

export dim
for D in (:InverseWishart, :LKJ, :MatrixBeta, :MatrixFDist, :Wishart)
    @eval begin
        @noinline function dim(d::$D)
            Base.depwarn("`dim(d::$($D))` is deprecated, use `size(d, 1)` instead.", :dim; force = true)
            size(d, 1)
        end
    end
end

# deprecated 12 September 2016
export circvar
@noinline function circvar(d)
    Base.depwarn("`circvar(d)` is deprecated, use `var(d)` instead.", :circvar; force = true)
    var(d)
end

# deprecated constructors with standard deviations
@noinline function MvNormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})
    Base.depwarn("`MvNormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})` is deprecated, use `MvNormal(μ, LinearAlgebra.Diagonal(map(abs2, σ)))` instead.", :MvNormal; force = true)
    MvNormal(μ, Diagonal(map(abs2, σ)))
end
@noinline function MvNormal(μ::AbstractVector{<:Real}, σ::Real)
    Base.depwarn("`MvNormal(μ::AbstractVector{<:Real}, σ::Real)` is deprecated, use `MvNormal(μ, σ^2 * LinearAlgebra.I)` instead.", :MvNormal; force = true)
    MvNormal(μ, σ^2 * I)
end
@noinline function MvNormal(σ::AbstractVector{<:Real})
    Base.depwarn("`MvNormal(σ::AbstractVector{<:Real})` is deprecated, use `MvNormal(LinearAlgebra.Diagonal(map(abs2, σ)))` instead.", :MvNormal; force = true)
    MvNormal(Diagonal(map(abs2, σ)))
end
@noinline function MvNormal(d::Int, σ::Real)
    Base.depwarn("`MvNormal(d::Int, σ::Real)` is deprecated, use `MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(σ^2, d)))` instead.", :MvNormal; force = true)
    MvNormal(Diagonal(Fill(σ^2, d)))
end

# Deprecated constructors
@noinline function MvNormalCanon(h::AbstractVector{<:Real}, prec::AbstractVector{<:Real})
    Base.depwarn("`MvNormalCanon(h::AbstractVector{<:Real}, prec::AbstractVector{<:Real})` is deprecated, use `MvNormalCanon(h, LinearAlgebra.Diagonal(prec))` instead.", :MvNormalCanon; force = true)
    MvNormalCanon(h, Diagonal(prec))
end
@noinline function MvNormalCanon(h::AbstractVector{<:Real}, prec::Real)
    Base.depwarn("`MvNormalCanon(h::AbstractVector{<:Real}, prec::Real)` is deprecated, use `MvNormalCanon(h, prec * LinearAlgebra.I)` instead.", :MvNormalCanon; force = true)
    MvNormalCanon(h, prec * I)
end
@noinline function MvNormalCanon(prec::AbstractVector)
    Base.depwarn("`MvNormalCanon(prec::AbstractVector)` is deprecated, use `MvNormalCanon(LinearAlgebra.Diagonal(prec))` instead.", :MvNormalCanon; force = true)
    MvNormalCanon(Diagonal(prec))
end
@noinline function MvNormalCanon(d::Int, prec::Real)
    Base.depwarn("`MvNormalCanon(d::Int, prec::Real)` is deprecated, use `MvNormalCanon(LinearAlgebra.Diagonal(FillArrays.Fill(prec, d)))` instead.", :MvNormalCanon; force = true)
    MvNormalCanon(Diagonal(Fill(prec, d)))
end

### Constructors of `Truncated` are deprecated - users should call `truncated`
export Truncated
@noinline function Truncated(d::UnivariateDistribution, l::Real, u::Real)
    Base.depwarn("`Truncated(d::UnivariateDistribution, l::Real, u::Real)` is deprecated, use `truncated(d, l, u)` instead.", :Truncated; force = true)
    truncated(d, l, u)
end
@noinline function Truncated(d::UnivariateDistribution, l::T, u::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real} 
    Base.depwarn("`Truncated(d::UnivariateDistribution, l::T, u::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real}` is deprecated, use `Truncated(d, l, u, log(lcdf), lcdf, ucdf, tp, logtp)` instead.", :Truncated; force = true)
    Truncated(d, l, u, log(lcdf), lcdf, ucdf, tp, logtp)
end

export Product
function Product(v::V) where {S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractVector{T}}
    Base.depwarn(
        "`Product(v::AbstractVector{<:UnivariateDistribution})` is deprecated, use `product_distribution(v)`",
        :Product;
        force = true,
    )
    return Product{S, T, V}(v)
end

export LocationScale
const LocationScale{T,S,D} = AffineDistribution{T,S,D}
function LocationScale(μ::Real, σ::Real, ρ::UnivariateDistribution; check_args::Bool=true)
    Base.depwarn(
        "`LocationScale(μ::Real, σ::Real, ρ::UnivariateDistribution; check_args::Bool=true)` is deprecated, use `μ + σ * d` instead.",
        :LocationScale;
        force = true,
    )
    # preparation for future PR where I remove σ > 0 check
    @check_args LocationScale (σ, σ > zero(σ))
    return AffineDistribution(μ, σ, ρ; check_args=false)
end

