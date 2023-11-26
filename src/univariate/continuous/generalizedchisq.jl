using Distributions, Random # remove
import Distributions: sampler, cf, cdf, pdf, quadgk, quantile_newton, quantile_bisect # remove

"""
    GeneralizedChisq(w, ν, λ, μ, σ)

The *Generalized chi-squared distribution* is the distribution of a sum of independent noncentral chi-squared variables and a normal variable:

```math
\\xi =\\sum_{i}w_{i}y_{i}+x,\\quad y_{i}\\sim \\chi '^{2}(\\nu_{i},\\lambda _{i}),\\quad x\\sim N(\\mu,\\sigma^{2}).
```

```julia
GeneralizedChisq(w, ν, λ, μ, σ)

```

External links

* [Generalized chi-squared distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_chi-squared_distribution)

"""
struct GeneralizedChisq{T<:Real, V<:AbstractVector{<:Real}} <: ContinuousUnivariateDistribution
    w::V
    ν::V
    λ::V
    μ::T
    σ::T

    function GeneralizedChisq{T, V}(w, ν, λ, μ, σ; check_args::Bool=true) where {T, V}
        # @check_args GeneralizedChisq (ν, all(ν .> 0)) (σ, σ ≥ zero(σ)) (length(w) == length(ν) == length(λ))
        new{T, V}(w, ν, λ, μ, σ)
    end
end

# Manage parameter types, ensuring that the parameters of normal are floats.

function GeneralizedChisq(w, ν, λ, μ::Tμ, σ::Tσ; check_args...) where {Tμ<:Real, Tσ<:Real}
    V = promote_type(typeof(w), typeof(ν), typeof(λ))
    T = promote_type(Tμ, Tσ, eltype(w), eltype(ν), eltype(λ))
    GeneralizedChisq{T, V}(w, ν, λ, μ, σ...; check_args...)
end

GeneralizedChisq(w, ν, λ, μ::Integer, σ::Integer; check_args...) = GeneralizedChisq(w, ν, λ, float(μ), float(σ); check_args...)

Base.eltype(::GeneralizedChisq{T,V}) where {T,V} = T

"""
    GeneralizedChisqSampler

Sampler of a generalized chi-squared distribution,
created by `sampler(::GeneralizedChisq)`.
"""
struct GeneralizedChisqSampler{T<:Real, SC<:Sampleable{Univariate, Continuous}, SN<:Sampleable{Univariate, Continuous}} <: Sampleable{Univariate, Continuous}
    μ::T
    nchisqsamplers::Vector{Tuple{T, SC}}
    normalsampler::SN # (zero-mean, since μ is defined apart)
    skipnormal::Bool  # Normal needs not be sampled if σ == 0
end

## Required functions

# sampler that predefines the distributions for batch sampling
function sampler(d::GeneralizedChisq{T}) where T
    μ = d.μ
    nchisqsamplers = [(d.w[i], sampler(NoncentralChisq(d.ν[i], d.λ[i]))) for i in eachindex(d.w)]
    normalsampler = sampler(Normal(zero(T), d.σ)) # zero-mean
    skipnormal = iszero(d.σ)
    GeneralizedChisqSampler(μ, nchisqsamplers, normalsampler, skipnormal)
end

function rand(rng::AbstractRNG, s::GeneralizedChisqSampler)
    result = s.μ
    for (w, ncs) in s.nchisqsamplers
        result += w * rand(rng, ncs)
    end
    if !s.skipnormal
        result += rand(rng, s.normalsampler)
    end
    return result
end

rand(rng::AbstractRNG, d::GeneralizedChisq) = rand(rng, sampler(d))

# cdf algorithm derived from https://github.com/abhranildas/gx2, by Abhranil Das.
function cdf(d::GeneralizedChisq{T}, x::Real) where T
    # special cases
    if iszero(d.σ)
        (x < minimum(d)) && return zero(T)
        (x > maximum(d)) && return one(T)
        if all(==(first(d.w)), d.w)
            iszero(first(d.w)) && return cdf(Normal(d.μ, d.σ), x)
            nchisq = NoncentralChisq{T}(sum(d.ν), sum(d.λ))
            (first(d.w) > zero(T)) && return cdf(nchisq, (x - d.μ)/first(d.w))
            (first(d.w) < zero(T)) && return ccdf(nchisq, (x - d.μ)/first(d.w))
        end
    end
    # general case
    GChisqComputations.daviescdf(d, x)
end

function pdf(d::GeneralizedChisq{T}, x::Real) where T
    # special cases
    if iszero(d.σ)
        !insupport(d, x) && return zero(T)
        if all(==(first(d.w)), d.w)
            iszero(first(d.w)) && return pdf(Normal(d.μ, d.σ), x)
            nchisq = NoncentralChisq{T}(sum(d.ν), sum(d.λ))
            return pdf(nchisq, (x - d.μ)/first(d.w)) / abs(first(d.w))
        end
    end
    # general case
    GChisqComputations.daviespdf(d, x)
end

logpdf(d::GeneralizedChisq, x::Real) = log(pdf(d, x))

function quantile(d::GeneralizedChisq{T}, p::Real) where T
    if 0 < p < 1
        # search starting point meeting convergence criterion
        x0 = mean(d)
        error0, curv0, converges = GChisqComputations.newtonconvergence(d, p, x0)
        if !converges
            bracket = GChisqComputations.definebracket(d, p, x0, error0, curv0, sqrt(var(d)))
            x0 = GChisqComputations.searchnewtonconvergence(d, p, bracket...)
        end
        return quantile_newton(d, p, x0)
    elseif p == 0
        return minimum(d)
    elseif p == 1
        return maximum(d)
    else
        return T(NaN)
    end
end

function minimum(d::GeneralizedChisq{T}) where T
    d.σ > zero(T) || any(<(zero(T)), d.w) ? typemin(T) : d.μ
end

function maximum(d::GeneralizedChisq{T}) where T
    d.σ > zero(T) || any(>(zero(T)), d.w) ? typemax(T) : d.μ
end

insupport(d::GeneralizedChisq, x::Real) = minimum(d) ≤ x ≤ maximum(d)

## Recommended functions

mean(d::GeneralizedChisq) = d.μ + sum(d.w[i] * (d.ν[i] + d.λ[i]) for i in eachindex(d.w))
var(d::GeneralizedChisq) = d.σ^2 + 2 * sum(d.w[i]^2 * (d.ν[i] + 2*d.λ[i]) for i in eachindex(d.w))

# modes(d::GeneralizedChisq)
# mode(d::GeneralizedChisq)
# skewness(d::GeneralizedChisq)
# kurtosis(d::GeneralizedChisq, ::Bool)
# entropy(d::GeneralizedChisq, ::Real)
# mgf(d::GeneralizedChisq, ::Any)
cf(d::GeneralizedChisq, t) = GChisqComputations.cf_explicit(d, t)

module GChisqComputations
    import ..Normal, ..NoncentralChisq, ..GeneralizedChisq
    import ..cf, ..cdf, ..insupport
    import ..quadgk

    # Characteristic function - explicit formula
    function cf_explicit(d, t)
        arg = im * d.μ * t
        denom = Complex(one(d.μ) * one(t)) # unit with stable type in later operations
        for i in eachindex(d.w)
            arg += im * d.w[i] * d.λ[i] * t / (1 - 2im * d.w[i] * t)
            denom *= (1 - 2im * d.w[i] * t)^(d.ν[i]/2)
        end
        arg -= d.σ^2 * t^2 / 2
        return exp(arg) / denom
    end

    # Characteristic function - by inheritance
    function cf_inherit(d::GeneralizedChisq{T}, t) where T
        result = exp(im * d.μ * t)
        for i in eachindex(d.w)
            result *= cf(NoncentralChisq{T}(d.ν[i], d.λ[i]), d.w[i] * t)
        end
        if !iszero(d.σ)
            result *= cf(Normal(zero(d.μ), d.σ), t)
        end
        return result
    end

    #=
    Terms of the formula (13) in Davies (1980) to calculate
    the cdf of a generalized chi-squared distribution, as: 
        F(x) = 1/2 - 1/π *∫sin(θ)/(u*ρ)du

    where `θ` and `ρ` are the outputs of this function.

    Those terms are related to the characteristic function of the distribution as:
        exp(θ*im)/ρ = exp(-u*x*im)*cf(u) 

    They can be also used to calculate the pdf as:
        f(x) = 1/π *∫cos(θ)/ρ du

    And its derivative as:
        f'(x) = 1/π *∫u*sin(θ)/ρ du
    =#
    function daviesterms(d::GeneralizedChisq{T}, u, x) where {T<:Real}
        θ = -T(u*(x - d.μ))
        ρ = exp(T(u*d.σ)^2 / 2)
        for i in eachindex(d.w)
            wi, νi, λi = T(d.w[i]), T(d.ν[i]), T(d.λ[i])
            τ  = 1 + 4 * wi^2 * u^2
            θ += νi*atan(2*wi*u)/2 + (λi*wi*u)/τ
            ρ *= τ^(νi/4) * exp(2λi*wi^2*u^2/τ)
        end
        return θ, ρ
	end

    function daviescdf(d, x)
        atol = eps(eltype(d)) # uniform tolerance to avoid issues at cdf ≈ 0.5 (integral ≈ 0)
        integral, _ = quadgk(0, Inf; atol=atol) do u
            θ, ρ = daviesterms(d, u, x)
            return sin(θ)/(u*ρ)
        end
        return 1/2 - integral/π
    end

    function daviespdf(d, x)
        integral, _ = quadgk(0, Inf) do u
            θ, ρ = daviesterms(d, u, x)
            return cos(θ)/ρ
        end
        return integral/π
    end

    # derivative of pdf
    function daviesdpdf(d, x)
        integral, _ = quadgk(0, Inf) do u
            θ, ρ = daviesterms(d, u, x)
            return u*sin(θ)/ρ
        end
        return integral/π
    end

    # functions to look for starting point where
    # the Newton method for quantiles converges:
    # sign(F(x) - q) == sign(f'(x))

    # Convergence criterion:
    function newtonconvergence(d::GeneralizedChisq{T}, p, x) where T
        error = cdf(d, x) - T(p)
        # out of bounds
        iszero(d.σ) && !insupport(d, x) && return error, zero(T), false
        # general case
        curvature = daviesdpdf(d, x)
        converges = sign(error) == sign(curvature)
        return error, curvature, converges
    end

    # Bracket containing solution for q, with x as one of the ends
    function definebracket(d::GeneralizedChisq{T}, p, x, errorx, curvx, initialwidth) where T
        # set direction of span and initialize bracket
        span = (errorx < 0) ? abs(initialwidth) : -abs(initialwidth)
        xbis = x + span
        errorxbis, curvxbis, _ = newtonconvergence(d, p, xbis)
        # greedier search if the bracket does not contain the solution
        iterations = 0
        while sign(errorxbis) == sign(errorx)
            iterations += 1
            span *= 2*errorxbis / (errorx - errorxbis)
            x, errorx, curvx = xbis, errorxbis, curvxbis
            xbis = x + span
            errorxbis, curvxbis, _ = newtonconvergence(d, p, xbis)
        end
        @debug "Iterations to define bracket" iterations
        # bracket end out of bounds
        if !insupport(d, xbis)
            xbis = d.μ
        end
        return (x, xbis), (errorx, errorxbis), (curvx, curvxbis)
    end

    # Bisect bracket until convergence point is reached
    function searchnewtonconvergence(d, p, (x,xbis), (errorx, errorxbis), (curvx, curvxbis))
        iterations = 0
        while true
            iterations += 1
            # Assumes that the first point is *outside* the region of convergence
            if sign(errorxbis) == sign(curvxbis)
                @debug "Iterations to find convergence point" iterations
                return xbis
            end
            xmid = (x + xbis) / 2
            errorxmid, curvxmid, _ = newtonconvergence(d, p, xmid)
            # switch first point if the new end of the bracket is on the same side
            if sign(errorxmid) == sign(errorx)
                x, errorx, curvx = xbis, errorxbis, curvxbis
            end
            xbis, errorxbis, curvxbis = xmid, errorxmid, curvxmid
        end
    end

end
