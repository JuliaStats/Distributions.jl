function ChainRulesCore.rrule(::Type{Distributions.EachVariate{V}}, x::AbstractArray{<:Real}) where {V}
    y = Distributions.EachVariate{V}(x)
    size_x = size(x)
    function EachVariate_pullback(Δ)
        # TODO: Should we also handle `Tangent{<:EachVariate}`?
        Δ_out = reshape(mapreduce(vec, vcat, ChainRulesCore.unthunk(Δ)), size_x)
        return (ChainRulesCore.NoTangent(), Δ_out)
    end
    return y, EachVariate_pullback
end
