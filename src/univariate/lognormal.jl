immutable LogNormal <: ContinuousUnivariateDistribution
    meanlog::Float64
    sdlog::Float64
    function LogNormal(ml::Real, sdl::Real)
    	if sdl > 0.0
    		new(float64(ml), float64(sdl))
    	else
    		error("sdlog must be positive")
    	end
	end
end

LogNormal(ml::Real) = LogNormal(ml, 1.0)
LogNormal() = LogNormal(0.0, 1.0)

@_jl_dist_2p LogNormal lnorm

entropy(d::LogNormal) = 0.5 + 0.5 * log(2.0 * pi * d.sdlog^2) + d.meanlog

insupport(::LogNormal, x::Real) = zero(x) < x < Inf
insupport(::Type{LogNormal}, x::Real) = zero(x) < x < Inf

function kurtosis(d::LogNormal)
    return exp(4.0 * d.sdlog^2) + 2.0 * exp(3.0 * d.sdlog^2) +
           3.0 * exp(2.0 * d.sdlog^2) - 6.0
end

mean(d::LogNormal) = exp(d.meanlog + d.sdlog^2 / 2)

median(d::LogNormal) = exp(d.meanlog)

# mgf(d::LogNormal)
# cf(d::LogNormal)

modes(d::LogNormal) = [exp(d.meanlog - d.sdlog^2)]

function skewness(d::LogNormal)
    return (exp(d.sdlog^2) + 2.0) * sqrt(exp(d.sdlog^2) - 1.0)
end

function var(d::LogNormal)
	sigsq = d.sdlog^2
	return (exp(sigsq) - 1) * exp(2d.meanlog + sigsq)
end

function fit_mle{T <: Real}(::Type{LogNormal}, x::Array{T})
    lx = log(x)
    return LogNormal(mean(lx), std(lx))
end
