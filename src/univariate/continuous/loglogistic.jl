"""
    LogLogistic(θ, ϕ)

The *log logistic distribution* with scale `θ` and shape `ϕ` is the distribution of a random variable whose logarithm has a [`Logistic`](@ref) distribution. 
If ``X \\sim \\operatorname{Logistic}(\\theta, \\phi)`` then ``exp(X) \\sim \\operatorname{LogLogistic}(\\theta, \\phi)``. The probability density function is 

```math
f(x; \\theta, \\phi) = \\frac{(\\phi / \\theta)x/\\theta()^(\\phi - 1)}{(1 + (x/\\theta)^\\phi)^2}, \\theta > 0, \\phi > 0
```

```julia
LogLogistic()            # Log-logistic distribution with unit scale and shape π 
LogLogistic(θ)           # Log-logistic distribution with scale θ and shape π
LogLogistic(θ,ϕ)         # Log-logistic distribution with scale θ and shape ϕ

params(d)                # Get the parameters, i.e. (θ, ϕ)
scale(d)                 # Get the scale parameter, i.e. θ
shape(d)                 # Get the shape parameter, i.e. ϕ
```

External links

* [Log logistic distribution on Wikipedia](https://en.wikipedia.org/wiki/Log-logistic_distribution)

"""


struct LogLogistic{T<:Real} <: ContinuousUnivariateDistribution
    θ::T
    ϕ::T
    LogLogistic{T}(θ::T,ϕ::T) where {T} = new{T}(θ,ϕ)
end

function LogLogistic(θ::T, ϕ::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogLogistic, θ > zero(θ) && ϕ > zero(ϕ))
    return LogLogistic{T}(θ, ϕ)
end

LogLogistic(θ::Real, ϕ::Real) = LogLogistic(promote(θ,ϕ)...)
LogLogistic(θ::Integer, ϕ::Integer) = LogLogistic(float(θ), float(ϕ))
LogLogistic(θ::T) where {T<:Real} = LogLogistic(θ, 2.0)
LogLogistic() = LogLogistic(1.0, 2.0, check_args=false) 

@distr_support LogLogistic 0.0 Inf

#### Coversions
convert(::Type{LogLogistic{T}}, θ::S, ϕ::S) where {T <: Real, S <: Real} = LogLogistic(T(θ), T(ϕ))
convert(::Type{LogLogistic{T}}, d::LogLogistic{S}) where {T <: Real, S <: Real} = LogLogistic(T(d.θ), T(d.ϕ), check_args=false)

#### Parameters 

params(d::LogLogistic) = (d.θ, d.ϕ)
partype(::LogLogistic{T}) where {T} = T

#### Statistics 

median(d::LogLogistic) = d.θ
function mean(d::LogLogistic)
	if d.ϕ ≤ 1
        error("mean is defined only when ϕ > 1") 	
	end
	return d.θ*π/d.ϕ/sin(π/d.ϕ)
end

function mode(d::LogLogistic)
	if d.ϕ ≤ 1
		error("mode is defined only when ϕ > 1")
	end 
	return d.θ*((d.ϕ-1)/(d.ϕ+1))^(1/d.ϕ)
end

function var(d::LogLogistic)
	if d.ϕ ≤ 2
		erros("var is defined only when ϕ > 2") 
	end
    b = π/d.ϕ
	return d.θ^2 * (2*b/sin(2*b)-b^2/(sin(b))^2)
end


#### Evaluation
function pdf(d::LogLogistic, x::Real)
    if x ≤ zero(0)
        z = zero(x)
    else
        z = d.ϕ/d.θ * (x/d.θ)^(d.ϕ-1) / (1+(x/d.θ)^d.ϕ)^2
    end    
    return z
end

function logpdf(d::LogLogistic, x::Real)
    if x ≤ zero(0)
        z = log(zero(x))
    else
        z = log(d.ϕ) - log(d.θ) + (d.ϕ-1)*(log(x)-log(d.θ)) - 2*log(1+(x/d.θ)^d.ϕ)
    end
    return oftype(z, -Inf)
end

function cdf(d::LogLogistic, x::Real)
    y = (x/d.θ)^d.ϕ
    z = y / (y + 1)
    return x ≤ 0 ? 0 : z
end

function logcdf(d::LogLogistic, x::Real)
    y = (x/d.θ)^d.ϕ
	z = log(y) - log(y+1) 
	return x ≤ 0 ? -Inf : z
end

function ccdf(d::LogLogistic, x::Real)
    y = (x/d.θ)^d.ϕ
	z = 1 / (y + 1) 
	return x ≤ 0 ? 1 : z 
end

function logccdf(d::LogLogistic, x::Real)
    y = (x/d.θ)^d.ϕ
	z = - log(y+1)
	return x ≤ 0 ? 0 : z 
end


#### Sampling
function rand(rng::AbstractRNG, d::LogLogistic)
    u = rand(rng)
    r = u / (1 - u)
    return r^(1/d.ϕ)*d.θ
end
