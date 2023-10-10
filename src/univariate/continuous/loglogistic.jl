"""
    LogLogistic(α, β)

The *log logistic distribution* with scale `α` and shape `β` is the distribution of a random variable whose logarithm has a [`Logistic`](@ref) distribution. 
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
function mean(d::LogLogistic{T}) where T<:Real
	if d.β ≤ 1
        ArgumentError("mean is defined only when β > 1") 	
	end
	return d.α*π/d.β/sin(π/d.β)
end

function mode(d::LogLogistic{T}) where T<:Real
	if d.β ≤ 1
		ArgumentError("mode is defined only when β > 1")
	end 
	return d.α*((d.β-1)/(d.β+1))^(1/d.β)
end

function var(d::LogLogistic{T}) where T<:Real
	if d.β ≤ 2
		ArgumentError("var is defined only when β > 2") 
	end
    b = π/d.β
	return d.α^2 * (2*b/sin(2*b)-b^2/(sin(b))^2)
end


#### Evaluation
function pdf(d::LogLogistic{T}, x::Real) where T<:Real
    # use built-in impletation to evaluate the density 
    # of loglogistic at x 
    # Y = log(X)
    # Y ~ logistic(log(θ), 1/ϕ)
    x >= 0 ? pdf(Logistic(log(d.α), 1/d.β), log(x)) / x : zero(T)
end

function logpdf(d::LogLogistic{T}, x::Real) where T<:Real
    x >= 0 ? logpdf(Logistic(log(d.α), 1/d.β), log(x)) + log(x) : -T(Inf)
end

function cdf(d::LogLogistic{T}, x::Real) where T<:Real
    x >= 0 ? cdf(Logistic(log(d.α), 1/d.β), log(x)) : zero(T)
end

function logcdf(d::LogLogistic{T}, x::Real) where T<:Real
    x >= 0 ? logcdf(Logistic(log(d.α), 1/d.β), log(x)) : -T(Inf)
end

function ccdf(d::LogLogistic{T}, x::Real) where T<:Real
    x >= 0 ? ccdf(Logistic(log(d.α), 1/d.β), log(x)) : one(T)
end

function logccdf(d::LogLogistic{T}, x::Real) where T<:Real
    x >= 0 ? logccdf(Logistic(log(d.α), 1/d.β), log(x)) : zero(T)
end


#### Sampling
function rand(rng::AbstractRNG, d::LogLogistic)
    u = rand(rng)
    r = u / (1 - u)
    return r^(1/d.β)*d.α
end
