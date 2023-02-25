## Callable struct to fix type inference issues caused by captured values
struct LogPDFNegativeBinomialPullback{D,T<:Real}
    ∂r::T
    ∂p::T
end

function (f::LogPDFNegativeBinomialPullback{D})(Δ) where {D}
    Δr = Δ * f.∂r
    Δp = Δ * f.∂p
    Δd = ChainRulesCore.Tangent{D}(; r=Δr, p=Δp)
    return ChainRulesCore.NoTangent(), Δd, ChainRulesCore.NoTangent()
end

function ChainRulesCore.rrule(::typeof(logpdf), d::NegativeBinomial, k::Real)
    # Compute log probability (as in the definition of `logpdf(d, k)` above)
    r, p = params(d)
    z = StatsFuns.xlogy(r, p) + StatsFuns.xlog1py(k, -p)
    if iszero(k)
        Ω = z
        ∂r = oftype(z, log(p))
        ∂p = oftype(z, r/p)
    elseif insupport(d, k)
        Ω = z - log(k + r) - SpecialFunctions.logbeta(r, k + 1)
        ∂r = oftype(z, log(p) - inv(k + r) - SpecialFunctions.digamma(r) + SpecialFunctions.digamma(r + k + 1))
        ∂p = oftype(z, r/p - k / (1 - p))
    else
        Ω = oftype(z, -Inf)
        ∂r = oftype(z, NaN)
        ∂p = oftype(z, NaN)
    end

    # Define pullback
    logpdf_NegativeBinomial_pullback = LogPDFNegativeBinomialPullback{typeof(d),typeof(z)}(∂r, ∂p)

    return Ω, logpdf_NegativeBinomial_pullback
end
