"""
    Burr(k, c, λ)
The *Burr distribution*  with shapes `k` and `c` and scale `λ` has probability density function
```math
f(x; k, c, \\lambda) = \\frac{ck}{\\lambda} \\left(\\frac{x}{\\lambda}\\right)^{c-1} \\left[ 1+\\left(\\frac{x}{\\lambda}\\right)^{c} \\right]^{-k-1}, \\quad x > 0
```
```julia
Burr()                  # Burr distribution with k = 1, c = 1 and λ = 1, i.e. Burr(1, 1, 1)
Burr(k, c)              # Burr distribution with λ = 1, i.e. Burr(k, c, 1)
Burr(k, c, λ)           # Burr distribution with shape1 k, shape2 c and scale λ
params(d)        # Get the parameters, i.e. (k, c, λ)
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
Burr(k::T, c::T) where {T <: Real} = Burr(k, c, one(T))
Burr(k::Real, c::Real) = Burr(promote(k, c)...)
Burr(k::T) where {T <: Real} = Burr(k, one(T), one(T))
Burr(k::Real) = Burr(promote(k)...)
Burr() = Burr(1, 1, 1; check_args=false)

@distr_support Burr 0.0 Inf

#### Conversions

function convert(::Type{Burr{T}}, k::Real, c::Real, λ::Real) where {T<:Real}
    Burr(T(k), T(c), T(λ))
end
function convert(::Type{Burr{T}}, d::Burr{S}) where {T <: Real, S <: Real}
    Burr(T(d.k), T(d.c), T(d.λ), check_args=false)
end

#### Parameters

shape(d::Burr) = (d.k, d.c)
scale(d::Burr) = d.λ
params(d::Burr) = (d.k, d.c, d.λ)
partype(::Burr{T}) where {T<:Real} = T

#### Statistics

function burr_moment(d::Burr, g::Real) # The calculating moments: this should not be exported.
    k, c, λ = params(d)

    tmp1 = max(0, 1 + g/c)
    tmp2 = max(0, k - g/c)
    result = λ^g * gamma(tmp1) * gamma(tmp2) / gamma(k)
    return if (g + c)*(g - k*c) < 0
        result
    else
        oftype(result, Inf)
    end
end


function median(d::Burr)
    k, c, λ = params(d)

    return λ * (2^(1/k) - 1)^(1/c)
end

mean(d::Burr) = burr_moment(d, 1)

function mode(d::Burr)
    k, c, λ = params(d)

    cmax = max(c, 1)
    result = λ * ((cmax - 1)/(k*c + 1))^(1/c)
    return if c > 1
        result
    else
        zero(result)
    end
end

function var(d::Burr)

    result = burr_moment(d, 2) - (burr_moment(d, 1))^2

    return if isinf(burr_moment(d, 1))
        oftype(result, Inf)
    else
        result
    end
end

function skewness(d::Burr)

    g1 = burr_moment(d, 1)
    g2 = burr_moment(d, 2)
    g3 = burr_moment(d, 3)

    result = (g3 - 3g1 * g2 + 2g1^3) / (g2 - g1^2) ^ (3/2)

    return if isinf(burr_moment(d, 2))
        oftype(result, Inf)
    else
        result
    end
end

function kurtosis(d::Burr)

    g1 = burr_moment(d, 1)
    g2 = burr_moment(d, 2)
    g3 = burr_moment(d, 3)
    g4 = burr_moment(d, 4)

    result = (g4 - 4g1 * g3 + 6g2 * g1^2 - 3 * g1^4) / (g2 - g1^2)^2 - 3

    return if isinf(burr_moment(d, 3))
        oftype(result, Inf)
    else
        result
    end
end

# function entropy(d::Burr{T}) where T<:Real
# end

function quantile(d::Burr, p::Real)
    k, c, λ = params(d)

    return λ * ( (1-p)^(-1/k) - 1 )^(1/c)
end

#### Evaluation

function pdf(d::Burr, x::Real)
    if isinf(x)
        return 0
    elseif x > 0
        k, c, λ = params(d)
        return k*c/λ * (x/λ)^(c-1) * ( 1 + (x/λ)^c )^(-k-1)
    else
        return 0
    end
end

function logpdf(d::Burr, x::Real)
    if isinf(x)
        return -Inf
    elseif x > 0
        k, c, λ = params(d)
        return log(k) + log(c) - log(λ) + (c-1)*log(x) - (c-1)*log(λ) - (k+1)*log1p( (x/λ)^c )
    else
        return -Inf
    end
end

function cdf(d::Burr, x::Real)
    if x > 0
        k, c, λ = params(d)
        return 1 - ( 1 + (x/λ)^c )^(-k)
    else
        return 0
    end
end

function ccdf(d::Burr, x::Real)
    if x > 0
        k, c, λ = params(d)
        return ( 1 + (x/λ)^c )^(-k)
    else
        return 1
    end
end

function logcdf(d::Burr, x::Real)
    if x > 0
        k, c, λ = params(d)
        return log1p( - ( 1 + (x/λ)^c )^(-k) )
    else
        return -Inf
    end
end

function logccdf(d::Burr, x::Real)
    if x > 0
        k, c, λ = params(d)
        return -k * log1p( (x/λ)^c )
    else
        return 0
    end
end

#### Sampling
function rand(rng::AbstractRNG, d::Burr)
    # Generate a Float64 random number uniformly in (0,1].
    u = 1 - rand(rng)

    return quantile(d, u)
end