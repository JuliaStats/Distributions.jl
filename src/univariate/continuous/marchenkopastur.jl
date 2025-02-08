"""
    MarchenkoPastur(λ, σ)

The *Marchenko-Pastur distribution* with asymptotic ratio λ and scale parameter σ.
For λ > 1, the pdf does have a point mass and is not well defined.


```julia
MarchenkoPastur(lambda, sigma)    # MarchenkoPastur distribution with asymptotic ratio lambda and scale parameter sigma
MarchenkoPastur(lambda)           # MarchenkoPastur distribution with asymptotic ratio lambda and unit scale.

params(d)                         # Get the parameters of the distribution
mean(d)                           # Get the mean of the distribution
std(d)                            # Get the standard deviation of the distribution
```

External links

* [Marchenko-Pastur distribution on Wikipedia](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)
* [Marchenko-Pastur distribution on Wolfram](https://www.wolfram.com/language/11/random-matrices/marchenko-pastur-distribution.html)
"""
struct MarchenkoPastur{T <: Real} <: ContinuousUnivariateDistribution
    λ::T
    σ::T
    MarchenkoPastur{T}(λ::T, σ::T) where {T <: Real} = new{T}(λ, σ)
end

function MarchenkoPastur(λ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(MarchenkoPastur, σ > zero(σ) && λ > zero(λ))
    return MarchenkoPastur{T}(λ, σ)
end

MarchenkoPastur(λ::Real, σ::Real) = MarchenkoPastur(promote(λ, σ)...)
MarchenkoPastur(λ::Integer, σ::Integer) = MarchenkoPastur(float(λ), float(σ))
MarchenkoPastur(λ::T) where {T <: Real} = MarchenkoPastur(λ, one(T))
MarchenkoPastur() = MarchenkoPastur(0.0, 1.0, check_args = false)

Base.convert(::Type{MarchenkoPastur{T}}, λ::S, σ::S) where {T <: Real, S <: Real} =
    MarchenkoPastur(T(λ), T(σ))
Base.convert(::Type{MarchenkoPastur{T}}, d::MarchenkoPastur{S}) where {T <: Real, S <: Real} =
    MarchenkoPastur(T(d.λ), T(d.σ), check_args = false)

@distr_support MarchenkoPastur -Inf Inf

#### Parameters
params(d::MarchenkoPastur) = (d.λ, d.σ)
@inline partype(::MarchenkoPastur{T}) where {T<:Real} = T

Base.eltype(::Type{MarchenkoPastur{T}}) where {T} = T

mean(d::MarchenkoPastur) = d.σ^2
var(d::MarchenkoPastur) = d.λ * d.σ^4
std(d::MarchenkoPastur) = sqrt(d.λ) * d.σ^2
skewness(d::MarchenkoPastur) = sqrt(d.λ)
kurtosis(d::MarchenkoPastur) = 2 + d.λ

# import HypergeometricFunctions._₂F₁
# moment(d::MarchenkoPastur, k) = d.σ^(2*k) * _₂F₁(1 - k, -k, 2, d.λ)

ratio(d::MarchenkoPastur) = d.λ
scale(d::MarchenkoPastur) = d.σ

Base.:*(x::Real, y::MarchenkoPastur)= MarchenkoPastur(y.λ, sqrt(x) * y.σ)
Base.:*(x::MarchenkoPastur, y::Real) = y * x

r(x, λp, λm) = sqrt((λp - x) / (x - λm))

function f(d, x, λp, λm)
    rx = r(x, λp, λm)
    λ, σ = d.λ, d.σ
    (π * λ +
     sqrt((λp - x) * (x - λm)) / σ^2 -
     (1 + λ) * atan(rx^2 - 1, 2 * rx) +
     (1 - λ) * atan(λm * rx^2 - λp, 2 * σ^2 * (1 - λ) * rx)) /
      (2 * π * λ)
end

# This implementation is from wikipedia and does not always match the Mathematica implementation
# function cdf(d::MarchenkoPastur{T}, x::Real) where T
#     λ, σ = d.λ, d.σ
#     λm = (σ * (1 - sqrt(λ)))^2
#     λp = (σ * (1 + sqrt(λ)))^2
#     if x < 0
#         zero(T)
#     elseif x >= λp
#         one(T)
#     else
#         if λ > 1
#             (x < λm ? (λ - 1) / λ : zero(T)) +
#                 (λm <= x < λp ? ((λ - 1) / (2 * λ) + f(d, x, λp, λm)) : zero(T))
#         else
#             (λm < x < λp ? f(d, x, λp, λm) : zero(T))
#         end
#     end
# end

# This is the implementation derived from Mathematicas output of
# CDF[MarchenkoPasturDistribution[\[Lambda], \[Sigma]], x]
function cdf(d::MarchenkoPastur{T}, x::Real) where T
    λ, σ = d.λ, d.σ
    λm = (σ * (1 - sqrt(λ)))^2
    λp = (σ * (1 + sqrt(λ)))^2
    xs = x / σ^2
    if λ < 1
        if λm < x < λp
            (π * λ +
             sqrt(4 * λ - (-1 - λ + xs)^2) -
             (1 + λ) * atan(1 + λ - xs,
                            sqrt(4 * λ - (-1 - λ + xs)^2)) -
             (1 - λ) * atan(-(-1 + λ)^2 + x * (1 + λ) / σ^2,
                            (1 - λ) * sqrt(4 * λ - (-1 - λ + xs)^2))) /
                                (2 * π * λ)
        elseif x >= λp
            one(T)
        else # x <= λm
            zero(T)
        end
    elseif λ == 1
        if λm < x < λp
            (π + sqrt(x * (4 - xs) / σ^2) -
             2 * atan(2 - xs,
                      sqrt(x * (4 - xs) / σ^2))) /
                          (2 * π)
        elseif x >= λp
            one(T)
        else # x <= λm
            zero(T)
        end
    elseif λ > 1
        if λm < x < λp
            1 + (-π + sqrt(4 * λ - (-1 - λ + xs)^2) -
                 (1 + λ) * atan(1 + λ - xs,
                                sqrt(4 * λ - (-1 - λ + xs)^2)) -
                 (-1 + λ) * atan(-(-1 + λ)^2 + x * (1 + λ) / σ^2,
                                 (-1 + λ) * sqrt(4 * λ - (-1 - λ + xs)^2))) /
                 (2 * π * λ)
        elseif 0 <= x <= λm
            1 - 1 / λ
        elseif x >= λp
            one(T)
        else # x <= λm
            zero(T)
        end
    else
        zero(T)
    end
end

function quantile(d::MarchenkoPastur{T}, q; ε = T(1e-10)) where T
    λ, σ = d.λ, d.σ
    l = T((σ * (1 - sqrt(λ)))^2)
    r = T((σ * (1 + sqrt(λ)))^2)
    x = (l + r) / 2
    while r - l > ε
        if cdf(d, x) > q
            r = x
        else
            l = x
        end
        x = (l + r) / 2
    end
    return x
end

median(d::MarchenkoPastur{T}; ε = T(1e-15)) where T = quantile(d, 0.5, ε = ε)

function pdf(d::MarchenkoPastur{T}, x::Real) where T
    λ, σ = d.λ, d.σ
    λm = (σ * (1 - sqrt(λ)))^2
    λp = (σ * (1 + sqrt(λ)))^2
    if λ > 1
        error("MarchenkoPastur distribution not well defined for λ > 1")
    else
        (λm <= x <= λp ? sqrt((λp - x) * (x - λm)) / (2 * π * d.σ^2 * d.λ * x) : zero(T))
    end
end
