immutable logNormal <: ContinuousUnivariateDistribution
    meanlog::Float64
    sdlog::Float64
    function logNormal(ml::Real, sdl::Real)
    	if sdl > 0.0
    		new(float64(ml), float64(sdl))
    	else
    		error("sdlog must be positive")
    	end
	end
end

logNormal(ml::Real) = logNormal(ml, 1.0)
logNormal() = logNormal(0.0, 1.0)

@_jl_dist_2p logNormal lnorm

entropy(d::logNormal) = 0.5 + 0.5 * log(2.0 * pi * d.sdlog^2) + d.meanlog

insupport(d::logNormal, x::Number) = isreal(x) && isfinite(x) && 0 < x

function kurtosis(d::logNormal)
    return exp(4.0 * d.sdlog^2) + 2.0 * exp(3.0 * d.sdlog^2) +
           3.0 * exp(2.0 * d.sdlog^2) - 6.0
end

mean(d::logNormal) = exp(d.meanlog + d.sdlog^2 / 2)

median(d::logNormal) = exp(d.meanlog)

# mgf(d::logNormal)
# cf(d::logNormal)

modes(d::logNormal) = [exp(d.meanlog - d.sdlog^2)]

function skewness(d::logNormal)
    return (exp(d.sdlog^2) + 2.0) * sqrt(exp(d.sdlog^2) - 1.0)
end

function var(d::logNormal)
	sigsq = d.sdlog^2
	return (exp(sigsq) - 1) * exp(2d.meanlog + sigsq)
end

function fit{T <: Real}(::Type{logNormal}, x::Array{T})
    lx = log(x)
    return logNormal(mean(lx), std(lx))
end
