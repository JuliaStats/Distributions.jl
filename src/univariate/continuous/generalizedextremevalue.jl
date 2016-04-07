immutable GeneralizedExtremeValue <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64
    ξ::Float64
    
    function GeneralizedExtremeValue(μ::Real, σ::Real, ξ::Real)
        σ > zero(σ) || error("Scale must be positive")
        new(μ, σ, ξ)
    end
end

minimum(d::GeneralizedExtremeValue) = d.ξ > 0.0 ? d.μ - d.σ / d.ξ : -Inf
maximum(d::GeneralizedExtremeValue) = d.ξ < 0.0 ? d.μ - d.σ / d.ξ : Inf


#### Parameters

shape(d::GeneralizedExtremeValue) = d.ξ
scale(d::GeneralizedExtremeValue) = d.σ
location(d::GeneralizedExtremeValue) = d.μ
params(d::GeneralizedExtremeValue) = (d.μ, d.σ, d.ξ)


#### Statistics
g(d::GeneralizedExtremeValue, k::Real) = gamma(1 - k * d.ξ) # This should not be exported. 

function median(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    
    if abs(ξ) < eps() # ξ == 0.0
        return μ - σ * log(log(2.0))
    else
        return μ + σ * (log(2.0) ^ (- ξ) - 1.0) / ξ
    end
end

function mean(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    
    if abs(ξ) < eps() # ξ == 0.0
        return μ + σ * γ
    elseif ξ < 1.0
        return μ + σ * (gamma(1.0 - ξ) - 1.0) / ξ
    else
        return Inf
    end
end

function mode(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    
    if abs(ξ) < eps() # ξ == 0.0
        return μ
    else
        return μ + σ * ((1.0 + ξ) ^ (- ξ) - 1.0) / ξ
    end
end

function var(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    
    if abs(ξ) < eps() # ξ == 0.0
        return σ ^ 2.0 * π ^ 2.0 / 6.0
    elseif ξ < 0.5
        return σ ^ 2.0 * (g(d, 2.0) - g(d, 1.0) ^ 2.0) / ξ ^ 2.0
    else
        return Inf
    end
end

function skewness(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    
    if abs(ξ) < eps() # ξ == 0.0
        return 12.0 * sqrt(6.0) * zeta(3.0) / pi ^ 3.0
    elseif ξ < 1.0 / 3.0
        g1 = g(d, 1)
        g2 = g(d, 2)
        g3 = g(d, 3)
        return sign(ξ) * (g3 - 3.0 * g1 * g2 + 2.0 * g1 ^ 3.0) / (g2 - g1 ^ 2.0) ^ (3.0 / 2.0)
    else
        return Inf
    end
end

function kurtosis(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    
    if abs(ξ) < eps() # ξ == 0.0
        return 12.0 / 5.0
    elseif ξ < 1.0 / 4.0
        g1 = g(d, 1)
        g2 = g(d, 2)
        g3 = g(d, 3)
        g4 = g(d, 4)
        return (g4 - 4.0 * g1 * g3 + 6.0 * g2 * g1 ^ 2.0 - 3.0 * g1 ^ 4.0) / (g2 - g1 ^ 2.0) ^ 2.0 - 3.0
    else
        return Inf
    end
end

function entropy(d::GeneralizedExtremeValue) 
    (μ, σ, ξ) = params(d)
    return log(σ) + γ * ξ + (1.0 + γ)
end

function quantile(d::GeneralizedExtremeValue, p::Float64)
    (μ, σ, ξ) = params(d)
	
    if abs(ξ) < eps() # ξ == 0.0
        return μ + σ * (- log(- log(p)))
    else
        return μ + σ * ((- log(p)) ^ (- ξ) - 1.0) / ξ
    end
end


#### Support 

insupport(d::GeneralizedExtremeValue, x::Real) = minimum(d) <= x <= maximum(d)


#### Evaluation

function logpdf(d::GeneralizedExtremeValue, x::Float64)
    if x == -Inf || x == Inf || ! insupport(d, x)
      return -Inf
    else
        (μ, σ, ξ) = params(d)
    
        z = (x - μ) / σ # Normalise x. 
        if abs(ξ) < eps() # ξ == 0.0
            t = z
            return - log(σ) - t - exp(- t)
        else
            if z * ξ == -1.0 # Otherwise, would compute zero to the power something. 
                return -Inf 
            else
                t = (1.0 + z * ξ) ^ (- 1.0 / ξ)
                return - log(σ) + (ξ + 1.0) * log(t) - t
            end
        end 
    end 
end

function pdf(d::GeneralizedExtremeValue, x::Float64)
    if x == -Inf || x == Inf || ! insupport(d, x)
        return 0.0
    else
        (μ, σ, ξ) = params(d)
    
        z = (x - μ) / σ # Normalise x. 
        if abs(ξ) < eps() # ξ == 0.0
            t = exp(- z)
            return (t * exp(- t)) / σ
        else
            if z * ξ == -1.0 # In this case: zero to the power something. 
                return 0.0
            else
                t = (1.0 + z * ξ) ^ (- 1.0 / ξ)
                return (t ^ (ξ + 1.0) * exp(- t)) / σ
            end
        end 
    end 
end

function logcdf(d::GeneralizedExtremeValue, x::Float64)
    if insupport(d, x)
        (μ, σ, ξ) = params(d)
    
        z = (x - μ) / σ # Normalise x. 
        if abs(ξ) < eps() # ξ == 0.0
            return - exp(- z)
        else
            return - (1.0 + z * ξ) ^ ( -1.0 / ξ)
        end
    elseif x <= minimum(d)
        return - Inf
    else
        return 0.0
    end
end

function cdf(d::GeneralizedExtremeValue, x::Float64)
    if insupport(d, x)
        (μ, σ, ξ) = params(d)
    
        z = (x - μ) / σ # Normalise x. 
        if abs(ξ) < eps() # ξ == 0.0
            t = exp(- z)
        else
            t = (1.0 + z * ξ) ^ (- 1.0 / ξ)
        end
        return exp(- t)
    elseif x <= minimum(d)
        return 0.0
    else
        return 1.0
    end
end

logccdf(d::GeneralizedExtremeValue, x::Float64) = log1p(- cdf(d, x))
ccdf(d::GeneralizedExtremeValue, x::Float64) = - expm1(logcdf(d, x))


#### Sampling

function rand(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    
    # Generate a Float64 random number uniformly in (0,1].
    u = 1.0 - rand()

    if abs(ξ) < eps() # ξ == 0.0
        rd = - log(- log(u))
    else
        rd = expm1(- ξ * log(- log(u))) / ξ
    end

    return μ + σ * rd
end
