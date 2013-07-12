immutable Poisson <: DiscreteUnivariateDistribution
    lambda::Float64
    function Poisson(l::Real)
    	if l > 0.0
    		new(float64(l))
    	else
    		error("lambda must be positive")
    	end
    end
end

Poisson() = Poisson(1.0)

@_jl_dist_1p Poisson pois

function entropy(d::Poisson)
    λ = d.lambda
    if λ < 50.0
        s = 0.0
        for k in 1:100
            s += λ^k * lgamma(k + 1.0) / gamma(k + 1.0)
        end
        return λ * (1.0 - log(λ)) + exp(-λ) * s
    else
        return 0.5 * log(2 * pi * e * λ) -
               (1 / (12 * λ)) -
               (1 / (24 * λ * λ)) -
               (19 / (360 * λ * λ * λ))
    end
end

insupport(d::Poisson, x::Number) = isinteger(x) && 0.0 <= x

kurtosis(d::Poisson) = 1.0 / d.lambda

mean(d::Poisson) = d.lambda

median(d::Poisson) = quantile(d, 0.5)

function mgf(d::Poisson, t::Real)
    l = d.lambda
    return exp(l * (exp(t) - 1.0))
end

function cf(d::Poisson, t::Real)
    l = d.lambda
    return exp(l * (exp(im * t) - 1.0))
end

modes(d::Poisson) = [floor(d.lambda)]

skewness(d::Poisson) = 1.0 / sqrt(d.lambda)

var(d::Poisson) = d.lambda

function fit_mle(::Type{Poisson}, x::Array)
    for i in 1:length(x)
        if !insupport(Poisson(), x[i])
            error("Poisson observations must be non-negative integers")
        end
    end
    Poisson(mean(x))
end
