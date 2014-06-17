# inverse transform sampler for discrete distributions
# efficient for small values, right-skewed distributions
immutable DiscreteITSampler{T<:AbstractVector} <: Sampler{Univariate,Discrete}
    values::T
    cdf::Vector{Float64}
end

function rand(s::DiscreteITSampler)
    u = rand()
    i = 0
    for i = 1:length(s.cdf)
        @inbounds if u <= s.cdf[i]
            return s.values[i]
        end
    end
    s.values[i+1]
end
