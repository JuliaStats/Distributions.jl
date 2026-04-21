"""
    LogLogistic(α, β)

The *log-logistic distribution* with scale ``\\alpha`` and shape ``\\beta`` is the distribution of a random variable whose logarithm has a [`Logistic`](@ref) distribution. 

If ``X \\sim \\operatorname{LogLogistic}(\\alpha, \\beta)`` then ``\\log(X) \\sim \\operatorname{Logistic}(\\log(\\alpha), 1/\\beta)``.
The probability density function is 

```math
f(x; \\alpha, \\beta) = \\frac{(\\beta / \\alpha){(x/\\alpha)}^{\\beta - 1}}{{(1 + {(x/\\alpha)}^\\beta)}^2}, \\qquad \\alpha > 0, \\beta > 0.
```

```julia
LogLogistic(α, β)        # Log-logistic distribution with scale α and shape β

params(d)                # Get the parameters, i.e. (α, β)
scale(d)                 # Get the scale parameter, i.e. α
shape(d)                 # Get the shape parameter, i.e. β
```

External links

* [Log-logistic distribution on Wikipedia](https://en.wikipedia.org/wiki/Log-logistic_distribution)
"""
struct LogLogistic{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    LogLogistic{T}(α::T,β::T) where {T} = new{T}(α,β)
end

function LogLogistic(α::T, β::T; check_args::Bool=true) where {T <: Real}
    check_args && @check_args(LogLogistic, α > zero(α) && β > zero(β))
    return LogLogistic{T}(α, β)
end

LogLogistic(α::Real, β::Real; check_args::Bool = true) = LogLogistic(promote(α, β)...; check_args)

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
        throw(ArgumentError("the mean of a log-logistic distribution is defined only when its shape β > 1"))
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
        throw(ArgumentError("the variance of a log-logistic distribution is defined only when its shape β > 2"))
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
    res = (β / _x) / ((1 + xoαβ) * (1 + inv(xoαβ)))
    return insupport ? res : zero(res)
end
function logpdf(d::LogLogistic, x::Real)
    (; α, β) = d
    insupport = x > 0
    _x = insupport ? x : zero(x)
    βlogxoα = β * log(_x / α)
    res = log(β / _x) - (log1pexp(βlogxoα) + log1pexp(-βlogxoα))
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

## Fitting

function _loglogistic_log_samples(x::AbstractArray{<:Real})
    isempty(x) && throw(ArgumentError("x cannot be empty."))

    logx = Vector{Float64}(undef, length(x))
    i = 0
    for xi in x
        i += 1
        xi > zero(xi) || throw(ArgumentError("LogLogistic fit requires all samples to be greater than zero."))
        logx[i] = log(float(xi))
    end

    return logx
end

function _loglogistic_mle_system(logx::AbstractVector{<:Real}, μ::Float64, φ::Float64)
    θ = exp(φ)
    invθ = inv(θ)
    invθsq = invθ^2

    gμ = 0.0
    gφ = 0.0
    hμμ = 0.0
    hμφ = 0.0
    hφφ = 0.0
    ll = 0.0

    for yi in logx
        z = (yi - μ) * invθ
        p = logistic(z)
        t = muladd(2.0, p, -1.0)
        s = 2.0 * p * (1.0 - p)

        gμ += t * invθ
        gφ += z * t - 1.0

        hμμ -= s * invθsq
        hμφ -= (t + z * s) * invθ
        hφφ -= z * t + z^2 * s

        u = -abs(z)
        ll += u - 2.0 * log1pexp(u) - φ
    end

    return ll, gμ, gφ, hμμ, hμφ, hφφ
end

"""
    fit_mle(::Type{<:LogLogistic}, x::AbstractArray{<:Real}; maxiter::Int=1000, tol::Real=1e-8)

Compute the maximum likelihood estimate of the [`LogLogistic`](@ref) distribution
by maximizing the logistic likelihood of `log.(x)` with Newton's method.
"""
function fit_mle(::Type{<:LogLogistic}, x::AbstractArray{<:Real};
    maxiter::Int = 1000, tol::Real = 1e-8)

    logx = _loglogistic_log_samples(x)

    μ = median(logx)
    q25, q75 = quantile(logx, (0.25, 0.75))
    θ = max((q75 - q25) / (2.0 * log(3.0)), sqrt(eps(Float64)))
    φ = log(θ)

    ll, gμ, gφ, hμμ, hμφ, hφφ = _loglogistic_mle_system(logx, μ, φ)

    converged = false
    t = 0
    while !converged && t < maxiter
        t += 1

        det = hμμ * hφφ - hμφ^2
        (!isfinite(det) || det <= 0.0) && break

        Δμ = (hφφ * gμ - hμφ * gφ) / det
        Δφ = (-hμφ * gμ + hμμ * gφ) / det

        step = 1.0
        accepted = false
        ll_old = ll

        while step > tol
            μ_new = μ - step * Δμ
            φ_new = φ - step * Δφ
            state_new = _loglogistic_mle_system(logx, μ_new, φ_new)
            ll_new = state_new[1]

            if isfinite(ll_new) && ll_new >= ll_old
                μ = μ_new
                φ = φ_new
                ll, gμ, gφ, hμμ, hμφ, hφφ = state_new
                accepted = true
                break
            end

            step /= 2.0
        end

        !accepted && break
        converged = max(abs(step * Δμ), abs(step * Δφ)) <= tol
    end

    return LogLogistic(exp(μ), exp(-φ))
end
