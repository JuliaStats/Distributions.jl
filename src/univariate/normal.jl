immutable Normal <: ContinuousUnivariateDistribution
    mean::Float64
    std::Float64
    function Normal(mu::Real, sd::Real)
    	sd > zero(sd) || error("std must be positive")
    	new(float64(mu), float64(sd))
    end
end

Normal(mu::Real) = Normal(mu, 1.0)
Normal() = Normal(0.0, 1.0)

const Gaussian = Normal

@_jl_dist_2p Normal norm

entropy(d::Normal) = 0.5 * log(2.0 * pi) + 0.5 + log(d.std)

insupport(d::Normal, x::Number) = isreal(x) && isfinite(x)

kurtosis(d::Normal) = 0.0

mean(d::Normal) = d.mean

median(d::Normal) = d.mean

function mgf(d::Normal, t::Real)
	m, s = d.mean, d.std
	return exp(t * m + 0.5 * s^t * t^2)
end

function cf(d::Normal, t::Real)
	m, s = d.mean, d.std
	return exp(im * t * m - 0.5 * s^t * t^2)
end

modes(d::Normal) = [d.mean]

rand(d::Normal) = d.mean + d.std * randn()

skewness(d::Normal) = 0.0

std(d::Normal) = d.std

var(d::Normal) = d.std^2

## Fit model

immutable NormalStats
    sx::Float64    # (weighted) sum of x
    sx2::Float64   # (weighted) sum of x^2
    tw::Float64    # total sample weight

    NormalStats(sx::Real, sx2::Real, tw::Real) = new(float64(sx), float64(sx2), float64(tw))
end

function suffstats(::Type{Normal}, x::Array) 
    sx = 0.
    sx2 = 0.
    for xi in x
        sx += xi
        sx2 += xi * xi
    end
    NormalStats(sx, sx2, length(x))
end

function suffstats(::Type{Normal}, x::Array, w::Array)
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    sx = 0.
    sx2 = 0.
    tw = 0.

    for i = 1:n
        xi = x[i]
        wi = w[i]
        sx += wi * xi
        sx2 += wi * (xi * xi)
        tw += wi
    end
    NormalStats(sx, sx2, tw)
end

function fit_mle(::Type{Normal}, ss::NormalStats)
    mu = ss.sx / ss.tw
    sig2 = ss.sx2 / ss.tw - mu * mu
    Normal(mu, sqrt(sig2))
end

function fit_mle{T<:Real}(::Type{Normal}, x::Array{T})
    n = length(x)
    mu = mean(x)
    sig2 = 0.
    for xi in x
        sig2 += abs2(xi - mu)
    end
    Normal(mu, sqrt(sig2 / n))
end

function fit_mle{T<:Real}(::Type{Normal}, x::Array{T}, w::Array{Float64})
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))        
    end

    sx = 0.
    tw = 0.

    for i = 1:n
        wi = w[i]
        sx += x[i] * wi
        tw += wi
    end
    mu = sx / tw

    sig2 = 0.
    for i = 1:n
        sig2 += abs2(x[i] - mu) * w[i]
    end
    sig2 /= tw

    Normal(mu, sqrt(sig2))
end


