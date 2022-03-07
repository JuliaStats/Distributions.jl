using StatsFuns: nchisqcdf

"""
    Skellam(μ1, μ2)

A *Skellam distribution* describes the difference between two independent [`Poisson`](@ref) variables, respectively with rate `μ1` and `μ2`.

```math
P(X = k) = e^{-(\\mu_1 + \\mu_2)} \\left( \\frac{\\mu_1}{\\mu_2} \\right)^{k/2} I_k(2 \\sqrt{\\mu_1 \\mu_2}) \\quad \\text{for integer } k
```

where ``I_k`` is the modified Bessel function of the first kind.

```julia
Skellam(μ1, μ2)     # Skellam distribution for the difference between two Poisson variables,
                    # respectively with expected values μ1 and μ2.

params(d)           # Get the parameters, i.e. (μ1, μ2)
```

External links:

* [Skellam distribution on Wikipedia](http://en.wikipedia.org/wiki/Skellam_distribution)
"""
struct Skellam{T<:Real} <: DiscreteUnivariateDistribution
    μ1::T
    μ2::T

    function Skellam{T}(μ1::T, μ2::T) where {T <: Real}
        return new{T}(μ1, μ2)
    end

end

function Skellam(μ1::T, μ2::T; check_args::Bool=true) where {T <: Real}
    @check_args Skellam (μ1, μ1 > zero(μ1)) (μ2, μ2 > zero(μ2))
    return Skellam{T}(μ1, μ2)
end

Skellam(μ1::Real, μ2::Real; check_args::Bool=true) = Skellam(promote(μ1, μ2)...; check_args=check_args)
Skellam(μ1::Integer, μ2::Integer; check_args::Bool=true) = Skellam(float(μ1), float(μ2); check_args=check_args)
function Skellam(μ::Real; check_args::Bool=true)
    @check_args Skellam (μ, μ > zero(μ))
    Skellam(μ, μ; check_args=false)
end
Skellam() = Skellam{Float64}(1.0, 1.0)

@distr_support Skellam -Inf Inf

#### Conversions

convert(::Type{Skellam{T}}, μ1::S, μ2::S) where {T<:Real, S<:Real} = Skellam(T(μ1), T(μ2))
Base.convert(::Type{Skellam{T}}, d::Skellam) where {T<:Real} = Skellam{T}(T(d.μ1), T(d.μ2))
Base.convert(::Type{Skellam{T}}, d::Skellam{T}) where {T<:Real} = d

#### Parameters

params(d::Skellam) = (d.μ1, d.μ2)
partype(::Skellam{T}) where {T} = T


#### Statistics

mean(d::Skellam) = d.μ1 - d.μ2

var(d::Skellam) = d.μ1 + d.μ2

skewness(d::Skellam) = mean(d) / (var(d)^(3//2))

kurtosis(d::Skellam) = 1 / var(d)


#### Evaluation

function logpdf(d::Skellam, x::Real)
    μ1, μ2 = params(d)
    if insupport(d, x)
        return - (μ1 + μ2) + (x/2) * log(μ1/μ2) + log(besseli(x, 2*sqrt(μ1)*sqrt(μ2)))
    else
        return one(x) / 2 * log(zero(μ1/μ2))
    end
end

function mgf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(μ1 * (exp(t) - 1) + μ2 * (exp(-t) - 1))
end

function cf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(μ1 * (cis(t) - 1) + μ2 * (cis(-t) - 1))
end

"""
    cdf(d::Skellam, t::Real)

Implementation based on SciPy: https://github.com/scipy/scipy/blob/v0.15.1/scipy/stats/_discrete_distns.py

Refer to Eqn (5) in On an Extension of the Connexion Between Poisson and χ2 Distributions, N.L Johnson(1959)
Vol 46, No 3/4, doi:10.2307/2333532
It relates the Skellam and Non-central chisquare PDFs, which is very similar to their CDFs computation as well.

Computing cdf of the Skellam distribution.
"""
function cdf(d::Skellam, t::Integer)
    μ1, μ2 = params(d)
    return if t < 0
        nchisqcdf(-2*t, 2*μ1, 2*μ2)
    else
        1 - nchisqcdf(2*(t+1), 2*μ2, 2*μ1)
    end
end

#### Sampling
rand(rng::AbstractRNG, d::Skellam) =
    rand(rng, Poisson(d.μ1)) - rand(rng, Poisson(d.μ2))
