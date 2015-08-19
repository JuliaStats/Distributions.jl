immutable GeneralizedPareto <: ContinuousUnivariateDistribution
    ξ::Float64
    σ::Float64
    μ::Float64

    function GeneralizedPareto(ξ::Real, σ::Real, μ::Real)
        @check_args(GeneralizedPareto, σ > zero(σ))
        new(ξ, σ, μ)
    end
    GeneralizedPareto(ξ::Real, σ::Real) = GeneralizedPareto(ξ::Real, σ::Real, 0.0)
    GeneralizedPareto() = new(1.0, 1.0, 0.0)
end

minimum(d::GeneralizedPareto) = d.μ
maximum(d::GeneralizedPareto) = d.ξ < 0.0 ? d.μ - d.σ / d.ξ : Inf


#### Parameters

shape(d::GeneralizedPareto) = d.ξ
scale(d::GeneralizedPareto) = d.σ
location(d::GeneralizedPareto) = d.μ
params(d::GeneralizedPareto) = (d.ξ, d.σ, d.μ)


#### Statistics

median(d::GeneralizedPareto) = d.μ + d.σ * expm1(d.ξ * log(2.0)) / d.ξ

function mean(d::GeneralizedPareto)
    if d.ξ < 1.0
        return d.μ + d.σ / (1.0 - d.ξ)
    else
        return Inf
    end
end

function var(d::GeneralizedPareto)
    if d.ξ < 0.5
        return d.σ^2 / ((1.0 - d.ξ)^2 * (1.0 - 2.0 * d.ξ))
    else
        return Inf
    end
end

function skewness(d::GeneralizedPareto)
    (ξ, σ, μ) = params(d)

    if ξ < (1.0 / 3.0)
        return 2.0 * (1.0 + ξ) * sqrt(1.0 - 2.0 * ξ) / (1.0 - 3.0 * ξ)
    else
        return Inf
    end
end

function kurtosis(d::GeneralizedPareto)
    (ξ, σ, μ) = params(d)

    if ξ < 0.25
        k1 = (1.0 - 2.0 * ξ) * (2.0 * ξ^2 + ξ + 3.0)
        k2 = (1.0 - 3.0 * ξ) * (1.0 - 4.0 * ξ)
        return 3.0 * k1 / k2 - 3.0
    else
        return Inf
    end
end


#### Evaluation

function logpdf(d::GeneralizedPareto, x::Float64)
    (ξ, σ, μ) = params(d)

    z = (x - μ) / σ
    p = 0.0
    if x >= μ
        if abs(ξ) < eps()
            p = -z - log(σ)
        elseif ξ > 0.0 || (ξ < 0.0 && x < maximum(d))
            p = (-1.0 - 1.0 / ξ) * log1p(z * ξ) - log(σ)
        end
    end

    return p
end

pdf(d::GeneralizedPareto, x::Float64) = exp(logpdf(d, x))

function logccdf(d::GeneralizedPareto, x::Float64)
    (ξ, σ, μ) = params(d)

    p = 1.0
    if x >= μ
        z = (x - μ) / σ
        if abs(ξ) < eps()
            p = -z
        elseif ξ > 0.0 || (ξ < 0.0 && x < maximum(d))
            p = (-1.0 / ξ) * log1p(z * ξ)
        end
    end

    return p
end

ccdf(d::GeneralizedPareto, x::Float64) = exp(logccdf(d, x))
cdf(d::GeneralizedPareto, x::Float64) = -expm1(logccdf(d, x))

function quantile(d::GeneralizedPareto, p::Float64)
    (ξ, σ, μ) = params(d)

    if p == 0.0
        z = 0.0
    elseif p == 1.0
        z = ξ < 0.0 ? -1.0 / ξ : Inf
    elseif 0.0 < p < 1.0
        if abs(ξ) < eps()
            z = -log1p(-p)
        else
            z = expm1(-ξ * log1p(-p)) / ξ
        end
    else
      z = NaN
    end

    return μ + σ * z
end


#### Sampling

function rand(d::GeneralizedPareto)
    # Generate a Float64 random number uniformly in (0,1].
    u = 1.0 - rand()

    if abs(d.ξ) < eps()
        rd = -log(u)
    else
        rd = expm1(-d.ξ * log(u)) / d.ξ
    end

    return d.μ + d.σ * rd
end
