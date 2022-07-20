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

"""
    support(d::MvDiscreteNonParametric)
Get a sorted AbstractVector defining the support of `d`.
"""
support(d::GeneralDiscreteNonParametric) = d.support

"""
    probs(d::MvDiscreteNonParametric)
Get the vector of probabilities associated with the support of `d`.
"""
probs(d::GeneralDiscreteNonParametric) = d.p


Base.length(d::GeneralDiscreteNonParametric) = length(first(d.support))

function _rand!(
    rng::AbstractRNG,
    d::GeneralDiscreteNonParametric,
    x::AbstractVector{T},
    ) where {T<:Real}

    length(x) == length(d) || throw(DimensionMismatch("Invalid argument dimension."))
    s = d.support
    p = d.p

    n = length(p)
    draw = Base.rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i+=1]
    end
    copyto!(x, s[i])
    return x
end


function _logpdf(d::GeneralDiscreteNonParametric, x::AbstractVector{T}) where {T<:Real}
    s = support(d)
    p = probs(d)
    for i = 1:length(p)
        if s[i] == x
            return log(p[i])
        end
    end
    return log(zero(eltype(p)))
end
