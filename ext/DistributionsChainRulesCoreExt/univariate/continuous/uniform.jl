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
