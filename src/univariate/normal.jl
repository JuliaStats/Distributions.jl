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
    s::Float64    # (weighted) sum of x
    μ::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight

    NormalStats(s::Real, u::Real, s2::Real, tw::Real) = new(float64(s), float64(u), float64(s2), float64(tw))
end

function suffstats{T<:Real}(::Type{Normal}, x::Array{T}) 
    n = length(x)

    # compute s
    s = x[1]
    for i = 2:n
        @inbounds s += x[i]
    end
    μ = s / n

    # compute ss
    s2 = abs2(x[1] - μ)
    for i = 2:n
        s2 += abs2(x[i] - μ)
    end

    NormalStats(s, μ, s2, n)
end

function suffstats{T<:Real}(::Type{Normal}, x::Array{T}, w::Array{Float64}) 
    n = length(x)

    # compute s
    tw = w[1]
    s = w[1] * x[1]
    for i = 2:n
        @inbounds wi = w[i] 
        @inbounds s += wi * x[i]
        tw += wi
    end
    μ = s / tw

    # compute ss
    s2 = w[1] * abs2(x[1] - μ)
    for i = 2:n
        @inbounds s2 += abs2(x[i] - μ)
    end

    NormalStats(s, μ, s2, tw)
end

fit_mle(::Type{Normal}, ss::NormalStats) = Normal(ss.μ, sqrt(ss.s2 / ss.tw))

