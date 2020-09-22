"""
    Burr(k, c, λ)
The *Burr distribution*  with shape1 `k`, shape2 `c` and scale `λ` has probability density function
```math
f(x; k, c, \\lambda) = \\frac{ck}{\\lambda} \\left(\\frac{x}{\\lambda}\\right)^{c-1} \\left[ 1+\\left(\\frac{x}{\\lambda}\\right)^{c} \\right]^{-k-1}, \\quad x > 0
```
```julia
Burr()                  # Burr distribution with k = 1, c = 1 and λ = 1, i.e. Burr(1, 1, 1)
Burr(k, c)              # Burr distribution with λ = 1, i.e. Burr(k, c, 1)
Burr(k, c, λ)           # Burr distribution with shape1 k, shape2 c and scale λ
params(d)        # Get the parameters, i.e. (k, c, λ)
shape1(d)        # Get the shape1 parameter, i.e. k
shape2(d)        # Get the shape2 parameter, i.e. c
scale(d)         # Get the scale parameter, i.e. λ
```
External links
* [Burr distribution on Wikipedia](https://en.wikipedia.org/wiki/Burr_distribution)
"""


struct Burr{T<:Real} <: ContinuousUnivariateDistribution
    k::T # shape1
    c::T # shape2
    λ::T # scale
end

function Burr(k::T, c::T, λ::T; check_args=true) where {T}
    check_args && @check_args(Burr, k > zero(k) && c > zero(c) && λ > zero(λ))
    return Burr{T}(k, c, λ)
end

Burr(k::Real, c::Real, λ::Real) = Burr(promote(k, c, λ)...)
Burr(k::Integer, c::Integer, λ::Integer) = Burr(float(k), float(c), float(λ))
Burr(k::T, c::T) where {T <: Real} = Burr(k, c, one(T))
Burr() = Burr(1.0, 1.0, 1.0, check_args=false)

@distr_support Burr 0.0 Inf

#### Conversions

function convert(::Type{Burr{T}}, k::S, c::S, λ::S) where {T <: Real, S <: Real}
    Burr(T(k), T(c), T(λ))
end
function convert(::Type{Burr{T}}, d::Burr{S}) where {T <: Real, S <: Real}
    Burr(T(d.k), T(d.c), T(d.λ), check_args=false)
end

#### Parameters

shape1(d::Burr) = d.k
shape2(d::Burr) = d.c
scale(d::Burr) = d.λ
params(d::Burr) = (d.k, d.c, d.λ)
partype(::Burr{T}) where {T<:Real} = T

#### Statistics

function m(d::Burr, g::Real) # The calculating moments: this should not be exported.
    (k, c, λ) = params(d)

    if (g-(-c))*(g-k*c) < 0
        λ^g * gamma(1 + g*c) * gamma(k - g/c) / gamma(k)
    else
        return T(Inf)
    end

end


function median(d::Burr{T}) where T<:Real
    (k, c, λ) = params(d)

    return λ * (2^(1/k) - 1)^(1/c)
end

function mean(d::Burr{T}) where T<:Real
    return m(d, 1)
end

function mode(d::Burr{T}) where T<:Real
    (k, c, λ) = params(d)

    if c > 1
        return λ * ( (c-1) / (k*c+1) )^(1/c)
    else
        return 0.0
    end
end

function var(d::Burr{T}) where T<:Real
    if isinf(m(d, 1))
        return T(Inf)
    else
        return m(d, 2) - (m(d, 1))^2
    end
end

function skewness(d::Burr{T}) where T<:Real
    if isinf(m(d, 2))
        return T(Inf)
    else
        g1 = m(d, 1)
        g2 = m(d, 2)
        g3 = m(d, 3)
        return (g3 - 3g1 * g2 + 2g1^3) / (g2 - g1^2) ^ (3/2)
    end
end

function kurtosis(d::Burr{T}) where T<:Real
    if isinf(m(d, 3))
        return T(Inf)
    else ξ < 1 / 4
        g1 = m(d, 1)
        g2 = m(d, 2)
        g3 = m(d, 3)
        g4 = m(d, 4)
        return (g4 - 4g1 * g3 + 6g2 * g1^2 - 3 * g1^4) / (g2 - g1^2)^2 - 3
    end
end

# function entropy(d::Burr{T}) where T<:Real
# end

function quantile(d::Burr, p::Real)
    (k, c, λ) = params(d)

    return λ * ( (1-p)^(-1/k) - 1 )^(1/c)
end

#### Evaluation

function pdf(d::Burr{T}, x::Real) where T<:Real
    if x > 0
        (k, c, λ) = params(d)
        return k*c/λ * (x/λ)^c * ( 1 + (x/λ)^c )^(-k-1)
    else
        return zero(T)
    end
end

function logpdf(d::Burr{T}, x::Real) where T<:Real
    if x > 0
        (k, c, λ) = params(d)
        return log(k) + log(c) - log(λ) + c*log(x) - c*log(λ) - (k+1)*log1p( (x/λ)^c )
    else
        return -T(Inf)
    end
end

function cdf(d::Burr{T}, x::Real) where T<:Real
    if x > 0
        (k, c, λ) = params(d)
        return 1 - ( 1 + (x/λ)^c )^(-k)
    else
        return zero(T)
    end
end

function ccdf(d::Burr{T}, x::Real) where T<:Real
    if x > 0
        (k, c, λ) = params(d)
        return ( 1 + (x/λ)^c )^(-k)
    else
        return one(T)
    end
end

function logcdf(d::Burr{T}, x::Real) where T<:Real
    if x > 0
        (k, c, λ) = params(d)
        return log1p( - ( 1 + (x/λ)^c )^(-k) )
    else
        return -T(Inf)
    end
end

function logccdf(d::Burr{T}, x::Real) where T<:Real
    if x > 0
        (k, c, λ) = params(d)
        return -k * log1p( (x/λ)^c )
    else
        return zero(T)
    end
end

#### Sampling
function rand(rng::AbstractRNG, d::Burr)
    # Generate a Float64 random number uniformly in (0,1].
    u = 1 - rand(rng)

    return quantile(d, u)
end