"""
    GeneralizedInverseGaussian(a, b, p)

The *generalized inverse Gaussian distribution* with parameters `a`, `b` and `p` has the probability density function

```math
f(x; a, b, p) = \\frac{(a/b)^{p/2}}{2 \\cdot \\operatorname{K}_{p}(\\sqrt{ab})} \\cdot x^{(p - 1)} \\cdot \\exp\\!( -ax/ 2) \\cdot \\exp\\!(b x^{-1}/2), \\quad x > 0
```

# TODO: Add julia examples of usage <29-04-23> 

Exernal links

* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)
"""
struct GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    p::T
    GeneralizedInverseGaussian{T}(a::T, b::T, p::T) where {T<:Real} = new{T}(a, b, p)
end

function GeneralizedInverseGaussian(a::T, b::T, p::T; check_args::Bool=true) where {T<:Real}
    @check_args GeneralizedInverseGaussian (a, a >= zero(a)) (b, b >= zero(b))
    if b == zero(b)
        return Gamma(p, 2 / a)
    elseif a == zero(a)
        return InverseGamma(-p, b / 2)
    end
    return GeneralizedInverseGaussian{T}(a, b, p)
end

GeneralizedInverseGaussian(a::Real, b::Real, p::Real; check_args::Bool=true) = GeneralizedInverseGaussian(promote(a, b, p)...; check_args=check_args)
GeneralizedInverseGaussian(a::Integer, b::Integer, p::Integer; check_args::Bool=true) = GeneralizedInverseGaussian(float(a), float(b), float(p); check_args=check_args)

# TODO: Default distributions for parameters not supplied <29-04-23> 

@distr_support GeneralizedInverseGaussian 0.0 Inf

#### Conversions

function convert(::Type{GeneralizedInverseGaussian{T}}, a::S, b::S, p::S) where {T<:Real,S<:Real}
    GeneralizedInverseGaussian(T(a), T(b), T(p))
end
function Base.convert(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian) where {T<:Real}
    GeneralizedInverseGaussian{T}(T(d.a), T(d.b), T(d.p))
end
Base.convert(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian{T}) where {T<:Real} = d

#### Parameters

params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
partype(::GeneralizedInverseGaussian{T}) where {T} = T

#### Statistics

function mean(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    r = sqrt(a * b)
    return sqrt(b / a) * besselk(p + 1, r) / besselk(p, r)
end

function var(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    r = sqrt(a * b)
    return (b / a) * (besselk(p + 2, r) / besselk(p, r) - (besselk(p + 1, r) / besselk(p, r))^2)
end

# TODO: skewness, kurtosis <29-04-23> 

function mode(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    ((p - 1) + sqrt((p - 1)^2 + a * b)) / a
end

# NOTE: Entropy could be implemented for integer and half integer values of `p` using tabulated values of the derivative
# at  https://dlmf.nist.gov/10.38#E7 <04-05-23> 

#### Evaluation

function pdf(d::GeneralizedInverseGaussian{T}, x::Real) where {T<:Real}
    if x > 0
        a, b, p = params(d)
        r = sqrt(a * b)
        return (a / b)^(p / 2) / (2 * besselk(p, r)) * x^(p - 1) * exp(-(a * x + b / x) / 2)
    else
        return zero(T)
    end
end

# TODO: logpdf, cdf, ccdf, logcdf, logccdf <29-04-23> 

@quantile_newton GeneralizedInverseGaussian

#### Sampling

# TODO: rand <29-04-23> 

#### Fit model

# TODO: SufficientStats, suffstats, fit_mle <29-04-23> 
