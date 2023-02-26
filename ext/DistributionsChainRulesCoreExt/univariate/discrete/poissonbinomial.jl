# Compute matrix of partial derivatives [∂P(X=j-1)/∂pᵢ]_{i=1,…,n; j=1,…,n+1}
#
# This implementation uses the same dynamic programming "trick" as for the computation of
# the primals.
#
# Reference (for the primal):
#
#      Marlin A. Thomas & Audrey E. Taub (1982)
#      Calculating binomial probabilities when the trial probabilities are unequal,
#      Journal of Statistical Computation and Simulation, 14:2, 125-131, DOI: 10.1080/00949658208810534
function poissonbinomial_pdf_partialderivatives(p::AbstractVector{<:Real})
    n = length(p)
    A = zeros(eltype(p), n, n + 1)
    @inbounds for j in 1:n
        A[j, end] = 1
    end
    @inbounds for (i, pi) in enumerate(p)
        qi = 1 - pi
        for k in (n - i + 1):n
            kp1 = k + 1
            for j in 1:(i - 1)
                A[j, k] = pi * A[j, k] + qi * A[j, kp1]
            end
            for j in (i+1):n
                A[j, k] = pi * A[j, k] + qi * A[j, kp1]
            end
        end
        for j in 1:(i-1)
            A[j, end] *= pi
        end
        for j in (i+1):n
            A[j, end] *= pi
        end
    end
    @inbounds for j in 1:n, i in 1:n
        A[i, j] -= A[i, j+1]
    end
    return A
end

for f in (:poissonbinomial_pdf, :poissonbinomial_pdf_fft)
    pullback = Symbol(f, :_pullback)
    @eval begin
        function ChainRulesCore.frule(
            (_, Δp)::Tuple{<:Any,<:AbstractVector{<:Real}}, ::typeof(Distributions.$f), p::AbstractVector{<:Real}
        )
            y = Distributions.$f(p)
            A = poissonbinomial_pdf_partialderivatives(p)
            return y, A' * Δp
        end
        function ChainRulesCore.rrule(::typeof(Distributions.$f), p::AbstractVector{<:Real})
            y = Distributions.$f(p)
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
