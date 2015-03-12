immutable InverseGaussian <: ContinuousUnivariateDistribution
    μ::Float64
    λ::Float64

    function InverseGaussian(μ::Real, λ::Real)
        (μ > zero(μ) && λ > zero(λ)) || error("InverseGaussian's μ and λ must be positive")
        @compat new(Float64(μ), Float64(λ))
    end

    InverseGaussian(μ::Real) = InverseGaussian(μ, 1.0)
    InverseGaussian() = new(1.0, 1.0)
end

@distr_support InverseGaussian 0.0 Inf


#### Parameters

shape(d::InverseGaussian) = d.λ
params(d::InverseGaussian) = (d.μ, d.λ)


#### Statistics

mean(d::InverseGaussian) = d.μ

var(d::InverseGaussian) = d.μ^3 / d.λ

skewness(d::InverseGaussian) = 3.0 * sqrt(d.μ / d.λ)

kurtosis(d::InverseGaussian) = 15.0 * d.μ / d.λ

function mode(d::InverseGaussian)
    μ, λ = params(d)
    r = μ / λ
    μ * (sqrt(1.0 + 2.25 * r^2) - 1.5 * r)
end


#### Evaluation

function pdf(d::InverseGaussian, x::Float64)
    if x > 0.0
        μ, λ = params(d)
        return sqrt(λ / (twoπ * x^3)) * exp(-λ * (x - μ)^2 / (2.0 * μ^2 * x))
    else
        return 0.0
    end 
end

function logpdf(d::InverseGaussian, x::Float64)
    if x > 0.0
        μ, λ = params(d)
        return 0.5 * (log(λ) - (log2π + 3.0 * log(x)) - λ * (x - μ)^2 / (μ^2 * x))
    else
        return -Inf
    end
end

function cdf(d::InverseGaussian, x::Float64)
    if x > 0.0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        return Φ(u * (v - 1.0)) + exp(2.0 * λ / μ) * Φ(-u * (v + 1.0))
    else
        return 0.0
    end
end

function ccdf(d::InverseGaussian, x::Float64)
    if x > 0.0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        Φc(u * (v - 1.0)) - exp(2.0 * λ / μ) * Φ(-u * (v + 1.0))
    else
        return 1.0
    end
end

function logcdf(d::InverseGaussian, x::Float64)
    if x > 0.0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        a = logΦ(u * (v -1.0)) 
        b = 2.0 * λ / μ + logΦ(-u * (v + 1.0))
        a + log1pexp(b - a)
    else
        return -Inf
    end
end

function logccdf(d::InverseGaussian, x::Float64)
    if x > 0.0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        a = logΦc(u * (v - 1.0)) 
        b = 2.0 * λ / μ + logΦ(-u * (v + 1.0))
        a + log1mexp(b - a)
    else
        return 0.0
    end
end

@quantile_newton InverseGaussian

#### Sampling

# rand method from:
#   John R. Michael, William R. Schucany and Roy W. Haas (1976) 
#   Generating Random Variates Using Transformations with Multiple Roots
#   The American Statistician , Vol. 30, No. 2, pp. 88-90
function rand(d::InverseGaussian)
    μ, λ = params(d)
    z = randn()
    v = z * z
    w = μ * v
    x1 = μ + μ / (2.0 * λ) * (w - sqrt(w * (4.0 * λ + w)))
    p1 = μ / (μ + x1)
    u = rand()
    u >= p1 ? μ^2 / x1 : x1
end

