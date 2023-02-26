for f in (:poissonbinomial_pdf, :poissonbinomial_pdf_fft)
    pullback = Symbol(f, :_pullback)
    @eval begin
        function ChainRulesCore.frule(
            (_, Δp)::Tuple{<:Any,<:AbstractVector{<:Real}}, ::typeof(Distributions.$f), p::AbstractVector{<:Real}
        )
            y = Distributions.$f(p)
            A = Distributions.poissonbinomial_pdf_partialderivatives(p)
            return y, A' * Δp
        end
        function ChainRulesCore.rrule(::typeof(Distributions.$f), p::AbstractVector{<:Real})
            y = Distributions.$f(p)
            A = Distributions.poissonbinomial_pdf_partialderivatives(p)
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
