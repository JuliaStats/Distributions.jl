#### naive sampling

struct CategoricalDirectSampler <: Sampleable{Univariate,Discrete}
    prob::Vector{Float64}

    function CategoricalDirectSampler(p::Vector{Float64})
        isempty(p) && throw(ArgumentError("p is empty."))
        new(p)
    end
end
ncategories(s::CategoricalDirectSampler) = length(s.prob)

function rand(s::CategoricalDirectSampler)
    p = s.prob
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end
    return i
end

