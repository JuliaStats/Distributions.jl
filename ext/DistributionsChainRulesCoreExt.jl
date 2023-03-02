module DistributionsChainRulesCoreExt

using Distributions
using Distributions: EachVariate, Dirichlet, Uniform, NegativeBinomial, _logpdf, poissonbinomial_pdf, poissonbinomial_pdf_fft,
                     xlogy, xlog1py, poissonbinomial_pdf_partialderivatives
using ChainRulesCore
using SpecialFunctions
using LinearAlgebra

ChainRulesCore.@non_differentiable Distributions.check_args(::Any, ::Bool)

function ChainRulesCore.rrule(::Type{EachVariate{V}}, x::AbstractArray{<:Real}) where {V}
    y = EachVariate{V}(x)
    size_x = size(x)
    function EachVariate_pullback(Δ)
        # TODO: Should we also handle `Tangent{<:EachVariate}`?
        Δ_out = reshape(mapreduce(vec, vcat, ChainRulesCore.unthunk(Δ)), size_x)
        return (ChainRulesCore.NoTangent(), Δ_out)
    end
    return y, EachVariate_pullback
end


# Dirichlet
function ChainRulesCore.frule((_, Δalpha)::Tuple{Any,Any}, ::Type{DT}, alpha::AbstractVector{T}; check_args::Bool = true) where {T <: Real, DT <: Union{Dirichlet{T}, Dirichlet}}
    d = DT(alpha; check_args=check_args)
    ∂alpha0 = sum(Δalpha)
    digamma_alpha0 = SpecialFunctions.digamma(d.alpha0)
    ∂lmnB = sum(Broadcast.instantiate(Broadcast.broadcasted(Δalpha, alpha) do Δalphai, alphai
        Δalphai * (SpecialFunctions.digamma(alphai) - digamma_alpha0)
    end))
    Δd = ChainRulesCore.Tangent{typeof(d)}(; alpha=Δalpha, alpha0=∂alpha0, lmnB=∂lmnB)
    return d, Δd
end

function ChainRulesCore.rrule(::Type{DT}, alpha::AbstractVector{T}; check_args::Bool = true) where {T <: Real, DT <: Union{Dirichlet{T}, Dirichlet}}
    d = DT(alpha; check_args=check_args)
    digamma_alpha0 = SpecialFunctions.digamma(d.alpha0)
    function Dirichlet_pullback(_Δd)
        Δd = ChainRulesCore.unthunk(_Δd)
        Δalpha = Δd.alpha .+ Δd.alpha0 .+ Δd.lmnB .* (SpecialFunctions.digamma.(alpha) .- digamma_alpha0)
        return ChainRulesCore.NoTangent(), Δalpha
    end
    return d, Dirichlet_pullback
end

function ChainRulesCore.frule((_, Δd, Δx)::Tuple{Any,Any,Any}, ::typeof(_logpdf), d::Dirichlet, x::AbstractVector{<:Real})
    Ω = _logpdf(d, x)
    ∂alpha = sum(Broadcast.instantiate(Broadcast.broadcasted(Δd.alpha, Δx, d.alpha, x) do Δalphai, Δxi, alphai, xi
        xlogy(Δalphai, xi) + (alphai - 1) * Δxi / xi
    end))
    ∂lmnB = -Δd.lmnB
    ΔΩ = ∂alpha + ∂lmnB
    if !isfinite(Ω)
        ΔΩ = oftype(ΔΩ, NaN)
    end
    return Ω, ΔΩ
end

function _logpdf_Dirichlet_∂alphai(xi, ΔΩi, isfinite::Bool)
    ∂alphai = xlogy.(ΔΩi, xi)
    return isfinite ? ∂alphai : oftype(∂alphai, NaN)
end

function _logpdf_Dirichlet_Δxi(ΔΩi, alphai, xi, isfinite::Bool)
    Δxi = ΔΩi * (alphai - 1) / xi
    return isfinite ? Δxi : oftype(Δxi, NaN)
end

function ChainRulesCore.rrule(::typeof(_logpdf), d::T, x::AbstractVector{<:Real}) where {T<:Dirichlet}
    Ω = _logpdf(d, x)
    isfinite_Ω = isfinite(Ω)
    alpha = d.alpha
    function _logpdf_Dirichlet_pullback(_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(_ΔΩ)
        ∂alpha = _logpdf_Dirichlet_∂alphai.(x, ΔΩ, isfinite_Ω)
        ∂lmnB = isfinite_Ω ? -float(ΔΩ) : oftype(float(ΔΩ), NaN)
        Δd = ChainRulesCore.Tangent{T}(; alpha=∂alpha, lmnB=∂lmnB)
        Δx = _logpdf_Dirichlet_Δxi.(ΔΩ, alpha, x, isfinite_Ω)
        return ChainRulesCore.NoTangent(), Δd, Δx
    end
    return Ω, _logpdf_Dirichlet_pullback
end


# Uniform
function ChainRulesCore.frule((_, Δd, _), ::typeof(logpdf), d::Uniform, x::Real)
    # Compute log probability
    a, b = params(d)
    insupport = a <= x <= b
    diff = b - a
    Ω = insupport ? -log(diff) : log(zero(diff))

    # Compute tangent
    Δdiff = Δd.a - Δd.b
    ΔΩ = (insupport ? Δdiff : zero(Δdiff)) / diff

    return Ω, ΔΩ
end

function ChainRulesCore.rrule(::typeof(logpdf), d::Uniform, x::Real)
    # Compute log probability
    a, b = params(d)
    insupport = a <= x <= b
    diff = b - a
    Ω = insupport ? -log(diff) : log(zero(diff))

    # Define pullback
    function logpdf_Uniform_pullback(Δ)
        Δa = Δ / diff
        Δd = if insupport
            ChainRulesCore.Tangent{typeof(d)}(; a=Δa, b=-Δa)
        else
            ChainRulesCore.Tangent{typeof(d)}(; a=zero(Δa), b=zero(Δa))
        end
        return ChainRulesCore.NoTangent(), Δd, ChainRulesCore.ZeroTangent()
    end

    return Ω, logpdf_Uniform_pullback
end


# Negative binomial
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
    z = xlogy(r, p) + xlog1py(k, -p)
    if iszero(k)
        Ω = z
        ∂r = oftype(z, log(p))
        ∂p = oftype(z, r/p)
    elseif insupport(d, k)
        Ω = z - log(k + r) - logbeta(r, k + 1)
        ∂r = oftype(z, log(p) - inv(k + r) - digamma(r) + digamma(r + k + 1))
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


# Poisson Binomial



for f in (:poissonbinomial_pdf, :poissonbinomial_pdf_fft)
    pullback = Symbol(f, :_pullback)
    @eval begin
        function ChainRulesCore.frule(
            (_, Δp)::Tuple{<:Any,<:AbstractVector{<:Real}}, ::typeof($f), p::AbstractVector{<:Real}
        )
            y = $f(p)
            A = poissonbinomial_pdf_partialderivatives(p)
            return y, A' * Δp
        end
        function ChainRulesCore.rrule(::typeof($f), p::AbstractVector{<:Real})
            y = $f(p)
            A = poissonbinomial_pdf_partialderivatives(p)
            function $pullback(Δy)
                p̄ = ChainRulesCore.InplaceableThunk(
                    Δ -> LinearAlgebra.mul!(Δ, A, Δy, true, true),
                    ChainRulesCore.@thunk(A * Δy),
                )
                return ChainRulesCore.NoTangent(), p̄
            end
            return y, $pullback
        end
    end
end

end
