"""
    GeneralizedGamma(a,d,p)

The *GeneralizedGamma distribution* with parameter a, d and p has probability density
function

```math
f(x; a, d, p) = \\frac{p}{a^p\\Gamma(d/p)} x^{d-1} e^{-\\left(\\frac{x}{a}\\right)^p},
\\quad x > 0
```

```julia
GeneralizedGamma()          # GeneralizedGamma distribution GeneralizedGamma(1, 1, 1)
GeneralizedGamma(a)         # GeneralizedGamma distribution GeneralizedGamma(a, 1, 1)

params(d)        # Get the parameters, i.e. (a, d, p)

S = rand(GeneralizedGamma(2, 3, 4), 5000); ##Sample from GeneralizedGamma(2, 3, 4)
fit_mle(GeneralizedGamma, S)  #MLE
```

External links

* [Generalized gamma distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_gamma_distribution)
"""

struct GeneralizedGamma{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    d::T
    p::T
    GeneralizedGamma{T}(a, d, p) where {T} = new{T}(a, d, p)
end

function GeneralizedGamma(a::T, d::T, p::T; check_args = true) where {T<:Real}
    check_args &&
        @check_args(GeneralizedGamma, a > zero(a) && d > zero(d) && p > zero(p))
    return GeneralizedGamma{T}(a, d, p)
end

GeneralizedGamma(a::Real, d::Real, p::Real) =
    GeneralizedGamma(promote(a, d, p)...)
GeneralizedGamma(a::Integer, d::Integer, p::Integer) =
    GeneralizedGamma(float(a), float(d), float(p))
GeneralizedGamma(a::T) where {T<:Real} = GeneralizedGamma(a, one(T), one(T))
GeneralizedGamma() = GeneralizedGamma(1.0, 1.0, 1.0, check_args = false)

@distr_support GeneralizedGamma 0.0 Inf

#### Conversions
convert(::Type{GeneralizedGamma{T}}, a::S, d::S, p::S) where {T<:Real,S<:Real} =
    GeneralizedGamma(T(a), T(d), T(p))
convert(
    ::Type{GeneralizedGamma{T}},
    d::GeneralizedGamma{S},
) where {T<:Real,S<:Real} =
    GeneralizedGamma(T(d.a), T(d.d), T(d.p), check_args = false)



#### Parameters

#shape(d::GeneralizedGamma) = d.p ##or d.d??
#scale(d::GeneralizedGamma) = d.a
#rate(d::GeneralizedGamma) = 1 / d.a

params(d::GeneralizedGamma) = (d.a, d.d, d.p)
partype(::GeneralizedGamma{T}) where {T} = T

#### Statistics

function mean(d::GeneralizedGamma)
    ip = 1 / d.p
    d.a * gamma((d.d + 1) * ip) / gamma(d.d * ip)
end
function var(d::GeneralizedGamma)
    ip = 1 / d.p
    iΓdp = 1 / gamma(d.d * ip)
    Γd1p = gamma((d.d + 1) * ip)
    Γd2p = gamma((d.d + 2) * ip)
    d.a^2 * (Γd2p * iΓdp - (Γd1p * iΓdp)^2)
end

function skewness(d::GeneralizedGamma)
    (a, d, p) = params(d)
    ip = 1 / p
    iΓdp = 1 / gamma(d * ip)
    Γd1p = gamma((d + 1) * ip)
    Γd2p = gamma((d + 2) * ip)
    Γd3p = gamma((d + 3) * ip)

    μ = a * Γd1p * iΓdp
    σ2 = a^2 * (Γd2p * iΓdp - (Γd1p * iΓdp)^2)
    EX3 = a^3 * Γd3p * iΓdp

    (EX3 - 3 * μ * σ2 - μ^3) / sqrt(σ2)^3
end

function kurtosis(d::GeneralizedGamma)
    (a, d, p) = params(d)
    ip = 1 / p
    iΓdp = 1 / gamma(d * ip)
    Γd1p = gamma((d + 1) * ip)
    Γd2p = gamma((d + 2) * ip)
    Γd3p = gamma((d + 3) * ip)
    Γd4p = gamma((d + 4) * ip)

    μ = a * Γd1p * iΓdp
    EX2 = a^2 * Γd2p * iΓdp
    EX3 = a^3 * Γd3p * iΓdp
    EX4 = a^4 * Γd4p * iΓdp
    (EX4 - 4 * EX3 * μ + 6 * EX2 * μ^2 - 3 * μ^4) / (EX2 - μ^2)^2 - 3
end

function mode(d::GeneralizedGamma)
    (a, d, p) = params(d)
    ip = 1 / p
    d >= 1 ? a * ((d - 1) * ip)^ip :
    error("GeneralizedGamma has no mode when d < 1")
end

function entropy(d::GeneralizedGamma)
    (a, d, p) = params(d)
    ip = 1 / p
    log(a * ip) + loggamma(d * ip) + d * ip - (d - 1) * ip * digamma(d * ip)
end

function logpdf(d::GeneralizedGamma{T}, x::Real) where {T<:Real}
    if x >= 0
        (a, d, p) = params(d)
        return log(p) - d * log(a) + (d - 1) * log(x) - (x / a)^p -
               loggamma(d / p)
    else
        return -T(Inf)
    end
end

pdf(d::GeneralizedGamma{T}, x::Real) where {T<:Real} = exp(logpdf(d, x))

# function cdf(d::GeneralizedGamma{T}, x::Real) where {T<:Real}
#     if x >= 0
#         (a, d, p) = params(d)
#         Γinc = gamma_inc(d / p, (x / a)^p, 0)
#         return Γinc[1] / (Γinc[1] + Γinc[2])
#     else
#         return zero(T)
#     end
# end
cdf(d::GeneralizedGamma{T}, x::Real) where {T<:Real} =
    cdf(Gamma(d.d / d.p, 1), (x / d.a)^d.p)
ccdf(d::GeneralizedGamma{T}, x::Real) where {T<:Real} =
    ccdf(Gamma(d.d / d.p, 1), (x / d.a)^d.p)

function quantile(d::GeneralizedGamma{T}, q::Real) where {T<:Real}
    (a, d, p) = params(d)
    ip = 1 / p
    a * quantile(Gamma(d * ip, 1), q)^ip
end

#function mgf(d::GeneralizedGamma{T}, t::Real) where {T<:Real}
#        g(x) = exp(logpdf(d, x) + t*x)
#        return quadgk(g, 0.0, Inf)
#end

# cf(d::GeneralizedGamma, t::Real)

function gradlogpdf(d::GeneralizedGamma{T}, x::Real) where {T<:Real}
    (a, d, p) = params(d)
    insupport(GeneralizedGamma, x) ? (-x^p * a^(-p) * p + d - 1) / x : zero(T)
end

#### Sampling
function rand(rng::AbstractRNG, d::GeneralizedGamma)
    (a, d, p) = params(d)
    if p == 1
        return rand(rng, Gamma(d, a))
    end
    ip = 1 / p
    return a * rand(rng, Gamma(d * ip, 1))^ip
end

#### Fit model
function fit_mle(
    ::Type{<:GeneralizedGamma},
    x::AbstractArray{T};
    maxiter::Int = 5000,
    tol::Float64 = 1e-6,
) where {T<:Real}
    n = length(x)
    logx = zeros(T, n)
    Σlogx = 0.0
    for i = 1:n
        logx[i] = log(x[i])
        Σlogx += logx[i]
    end
    ##Initial values
    pu, pl = 0.0, 0.0
    for i = 1.0:2.0:5000.0
        MM = gengamma_MLE_update_function(i, n, x, logx, Σlogx)
        if MM[4] < 0
            pu = i
            pl = i - 2

            break
        end
    end
    if gengamma_MLE_update_function(pu * 1.05, n, x, logx, Σlogx)[5]
        return error("K is zero.") # For some extreme conditions.
    end
    #print((pl,pu))
    if pu == 0
        return error("p is very large.")
    end
    pl = max(pl, 1.0e-7)
    # if isnan(gengamma_MLE_update_function(pl, n, x, logx, Σlogx)[4])
    #     return error("p is very small.")
    # end
    #if isnan(gengamma_MLE_update_function(pu, n, x, logx, Σlogx)[4])
    #    return error("p is very large.")
    #end
    a = 0.0
    d = 0.0
    p = 0.0
    p0 = (pl + pu) * 0.5
    for i = 1:maxiter
        #println(p0)
        if p0 < pl || p0 > pu || isnan(p0)
            p0 = pl + (pu - pl) * rand()
        end
        a, d, p = gengamma_MLE_update_function(p0, n, x, logx, Σlogx)
        if abs(p0 - p) <= tol
            return GeneralizedGamma(a, d, p)
        end
        p0 = p
    end
    @warn("Maximum iteration reached.")
    return GeneralizedGamma(a, d, p)
end

function gengamma_MLE_update_function(p, n, x, logx, Σlogx)
    Σxp = 0.0
    ∂Σxp = 0.0
    ∂2Σxp = 0.0
    ∂3Σxp = 0.0
    ip = 1 / p
    for i = 1:n
        xp = x[i]^p
        log_x = logx[i]
        Σxp += xp
        ∂Σxp += xp * log_x
        ∂2Σxp += xp * log_x^2
        ∂3Σxp += xp * log_x^3
    end
    K1 = n * ∂Σxp - Σlogx * Σxp
    K = 1 / K1
    ap = n^2 * ip * K
    a = ap^(-ip) ##exp(-ip*log(ap))
    log_ap = log(ap)
    log_a = -ip * log_ap
    ia = 1 / a
    d = n * Σxp * K
    dip = d * ip
    pΣxp = p * Σxp
    loggamma_dip = loggamma(d * ip)
    #ld = -n * log(a) + Σlogx - n * digamma_dip * ip
    ##Derivatives
    ∂ip = -ip^2
    ∂K1 = n * ∂2Σxp - Σlogx * ∂Σxp
    ∂K = -∂K1 * K^2
    ∂ap = n^2 * (∂ip * K + ip * ∂K)
    ∂log_ap = ∂ap / ap

    a_temp = -(∂ip * log_ap + ip * ∂log_ap)
    ∂a = a_temp * a
    ∂log_a = ∂a * ia
    #∂ia = (∂ip * log(ap) + ip * ∂ap / ap) * ia
    ∂d = n * (∂Σxp * K + Σxp * ∂K)
    #∂pΣxp = Σxp + p * ∂Σxp
    ∂dip = (∂d * ip + ∂ip * d)
    digamma_dip = digamma(dip)

    ∂loggamma_dip = ∂dip * digamma_dip
    #∂digamma_dip = trigamma(d * ip) * (∂d * ip + ∂ip * d)
    #∂ld = -n * ∂a * ia - n * (∂digamma_dip * ip + digamma_dip * ∂ip)
    ##Second derivatives
    ∂2ip = -2 * ∂ip * ip
    ∂2K1 = n * ∂3Σxp - Σlogx * ∂2Σxp
    ∂2K = -∂2K1 * K^2 - 2 * K * ∂K * ∂K1
    ∂2ap = n^2 * ((∂2ip * K + ∂ip * ∂K) + (∂ip * ∂K + ip * ∂2K))
    ∂2log_ap = (∂2ap * ap - ∂ap^2) / ap^2
    ∂a_temp =
        -((∂2ip * log_ap + ∂ip * ∂log_ap) + (∂ip * ∂log_ap + ip * ∂2log_ap))
    ∂2a = ∂a_temp * a + a_temp * ∂a
    ∂2log_a = (∂2a * a - ∂a^2) * ia^2

    ∂2d = n * ((∂2Σxp * K + ∂Σxp * ∂K) + (∂Σxp * ∂K + Σxp * ∂2K))
    ∂digamma_dip = ∂dip * trigamma(dip)
    ∂2dip = (∂2d * ip + ∂d * ∂ip + ∂2ip * d + ∂ip * ∂d)
    ∂2loggamma_dip = ∂2dip * digamma_dip + ∂dip * ∂digamma_dip
    ### log likelihood as a function of p. (a and d have closed form depend on p)
    llike =
        n * log(p) - n * d * log_a + (d - 1) * Σlogx - Σxp * ap -
        n * loggamma_dip

    ∂llike =
        n * ip - n * (∂d * log_a + d * ∂log_a) + ∂d * Σlogx -
        (∂Σxp * ap + ∂ap * Σxp) - n * ∂loggamma_dip

    ∂2llike =
        n * ∂ip - n * (∂2d * log_a + 2 * ∂d * ∂log_a + d * ∂2log_a) +
        ∂2d * Σlogx - (∂2Σxp * ap + 2 * ∂Σxp * ∂ap + Σxp * ∂2ap) -
        n * ∂2loggamma_dip

    return a, d, p - ∂llike / ∂2llike, ∂llike, ∂K == 0, llike
end

# Functions h(α) and h'(α). We will solve h(α) = 0 using Newton's method.
# α = p
# function gengamma_h_dh(
#     α::Float64,
#     n::Int64,
#     X::AbstractArray{T},
#     logX::AbstractArray{T},
#     slogX::Float64,
# ) where {T<:Real}
#     iα1 = 1 / (α + 1)
#     Sx = 0.0
#     Sxlog = 0.0
#     dSxlog = 0.0
#     for i = 1:n
#         xα = X[i]^α
#         Sx += xα
#         Sxlog += xα * logX[i]
#         dSxlog += xα * logX[i]^2
#     end
#     k = (n - 1.5 - α * iα1) / α - log(α) * α * iα1^2
#     dk = (α - 1) * log(α) * iα1^3 - (n - 1.5) / α^2
#     dSx = Sxlog
#     A = k * Sx
#     B = n * Sxlog - Sx * slogX
#     ϕ = A / B
#     dA = dk * Sx + k * dSx
#     dB = n * dSxlog - dSx * slogX
#     dϕ = (dA * B - A * dB) / B^2
#
#     H = n * digamma(n * ϕ) + α * slogX - 1 / ϕ - n * digamma(ϕ) - n * log(Sx)
#
#     dH =
#         n^2 * dϕ * trigamma(n * ϕ) + slogX + dϕ / ϕ^2 - n * dϕ * trigamma(ϕ) -
#         n * dSx / Sx
#     return H, dH, ϕ, Sx
# end

# # function sumlogpdf(x, n, slogx, a, d, p)
# #     a1 = 1 / a
# #     n * log(p) + (d - 1) * slogx - sum((x * a1) .^ p) + n * d * log(a1) -
# #     n * log(gamma(d / p))
# # end
