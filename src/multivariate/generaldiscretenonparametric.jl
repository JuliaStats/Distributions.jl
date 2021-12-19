struct GeneralDiscreteNonParametric{VF,T,P <: Real,Ts <: AbstractVector{T},Ps <: AbstractVector{P},} <: Distribution{VF,Discrete}
    support::Ts
    p::Ps

    function GeneralDiscreteNonParametric{VF,T,P,Ts,Ps}(
        support::Ts,
        p::Ps;
        check_args=true,
    ) where {VF,T,P <: Real,Ts <: AbstractVector{T},Ps <: AbstractVector{P}}
        if check_args
            length(support) == length(p) ||
                error("length of `support` and `p` must be equal")
            isprobvec(p) || error("`p` must be a probability vector")
            allunique(support) || error("`support` must contain only unique values")
        end
        new{VF,T,P,Ts,Ps}(support, p)
    end
end

function rand(rng::AbstractRNG, d::GeneralDiscreteNonParametric)
    x = support(d)
    p = probs(d)
    n = length(p)
    draw = rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i +=1]
    end
    return x[i]
end
