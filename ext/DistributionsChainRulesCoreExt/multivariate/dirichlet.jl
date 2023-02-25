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

function ChainRulesCore.frule((_, Δd, Δx)::Tuple{Any,Any,Any}, ::typeof(Distributions._logpdf), d::Dirichlet, x::AbstractVector{<:Real})
    Ω = Distributions._logpdf(d, x)
    ∂alpha = sum(Broadcast.instantiate(Broadcast.broadcasted(Δd.alpha, Δx, d.alpha, x) do Δalphai, Δxi, alphai, xi
        StatsFuns.xlogy(Δalphai, xi) + (alphai - 1) * Δxi / xi
    end))
    ∂lmnB = -Δd.lmnB
    ΔΩ = ∂alpha + ∂lmnB
    if !isfinite(Ω)
        ΔΩ = oftype(ΔΩ, NaN)
    end
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(::typeof(Distributions._logpdf), d::T, x::AbstractVector{<:Real}) where {T<:Dirichlet}
    Ω = Distributions._logpdf(d, x)
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
function _logpdf_Dirichlet_∂alphai(xi, ΔΩi, isfinite::Bool)
    ∂alphai = StatsFuns.xlogy.(ΔΩi, xi)
    return isfinite ? ∂alphai : oftype(∂alphai, NaN)
end
function _logpdf_Dirichlet_Δxi(ΔΩi, alphai, xi, isfinite::Bool)
    Δxi = ΔΩi * (alphai - 1) / xi
    return isfinite ? Δxi : oftype(Δxi, NaN)
end
