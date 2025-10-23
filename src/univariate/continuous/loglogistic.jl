"""
    LogLogistic(α, β)

The *log-logistic distribution* with scale `α` and shape `β` is the distribution of a random variable whose logarithm has a [`Logistic`](@ref) distribution. 
If ``X \\sim \\operatorname{LogLogistic}(\\alpha, \\beta)`` then ``log(X) \\sim \\operatorname{Logistic}(log(\\alpha), 1/\\beta)``. The probability density function is 

```math
f(x; \\alpha, \\beta) = \\frac{(\\alpha / \\beta)x/\\beta()^(\\alpha - 1)}{(1 + (x/\\beta)^\\alpha)^2}, \\beta > 0, \\alpha > 0
```

```julia
LogLogistic()            # Log-logistic distribution with unit scale and unit shape
LogLogistic(α,β)         # Log-logistic distribution with scale α and shape β

params(d)                # Get the parameters, i.e. (α, β)
scale(d)                 # Get the scale parameter, i.e. α
shape(d)                 # Get the shape parameter, i.e. β
```

External links

* [Log logistic distribution on Wikipedia](https://en.wikipedia.org/wiki/Log-logistic_distribution)
"""
struct LogLogistic{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    LogLogistic{T}(α::T,β::T) where {T} = new{T}(α,β)
end

function LogLogistic(α::T, β::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogLogistic, α > zero(α) && β > zero(β))
    return LogLogistic{T}(α, β)
end

LogLogistic(α::Real, β::Real) = LogLogistic(promote(α,β)...)
LogLogistic(α::Integer, β::Integer) = LogLogistic(float(α), float(β))
LogLogistic() = LogLogistic(1.0, 1.0, check_args=false) 

@distr_support LogLogistic 0.0 Inf

#### Coversions
convert(::Type{LogLogistic{T}}, d::LogLogistic{T}) where {T<:Real} = d
convert(::Type{LogLogistic{T}}, d::LogLogistic) where {T<:Real} = LogLogistic{T}(T(d.α), T(d.β))

#### Parameters
params(d::LogLogistic) = (d.α, d.β)
partype(::LogLogistic{T}) where {T} = T

#### Statistics 

median(d::LogLogistic) = d.α
function mean(d::LogLogistic)
    (; α, β) = d
	if !(β > 1)
        ArgumentError("the mean of a log-logistic distribution is defined only when its shape β > 1") 	
	end
    return α/sinc(inv(β))
end

function mode(d::LogLogistic)
    (; α, β) = d
    return α*(max(β - 1, 0) / (β + 1))^inv(β)
end

function var(d::LogLogistic)
    (; α, β) = d
	if !(β > 2)
        ArgumentError("the variance of a log-logistic distribution is defined only when its shape β > 2") 	
	end
    invβ = inv(β)
	return α^2 * (inv(sinc(2 * invβ)) - inv(sinc(invβ))^2)
end

entropy(d::LogLogistic) = log(d.α / d.β) + 2

#### Evaluation

function pdf(d::LogLogistic, x::Real)
    (; α, β) = d
    insupport = x > 0
    _x = insupport ? x : zero(x)
    xoαβ = (_x / α)^β
    res = (β / x) / ((1 + xoαβ) * (1 + inv(xoαβ)))
    return insupport ? res : zero(res)
end
function logpdf(d::LogLogistic, x::Real)
    (; α, β) = d
    insupport = x > 0
    _x = insupport ? x : zero(x)
    βlogxoα = β * log(_x / α)
    res = log(β / x) - (log1pexp(βlogxoα) + log1pexp(-βlogxoα))
    return insupport ? res : oftype(res, -Inf)
end

cdf(d::LogLogistic, x::Real) = inv(1 + (max(x, 0) / d.α)^(-d.β))
ccdf(d::LogLogistic, x::Real) = inv(1 + (max(x, 0) / d.α)^d.β)

logcdf(d::LogLogistic, x::Real) = -log1pexp(-d.β * log(max(x, 0) / d.α))
logccdf(d::LogLogistic, x::Real) = -log1pexp(d.β * log(max(x, 0) / d.α))

quantile(d::LogLogistic, p::Real) = d.α * (p / (1 - p))^inv(d.β)
cquantile(d::LogLogistic, p::Real) = d.α * ((1 - p) / p)^inv(d.β)

invlogcdf(d::LogLogistic, lp::Real) = d.α * expm1(-lp)^(-inv(d.β))
invlogccdf(d::LogLogistic, lp::Real) = d.α * expm1(-lp)^inv(d.β)

#### Sampling

function rand(rng::AbstractRNG, d::LogLogistic)
    u = rand(rng)
    return d.α * (u / (1 - u))^(inv(d.β))
end
function rand!(rng::AbstractRNG, d::LogLogistic, A::AbstractArray{<:Real})
    rand!(rng, A)
    let α = d.α, invβ = inv(d.β)
        map!(A, A) do u
            return α * (u / (1 - u))^invβ
        end
    end
    return A
end
