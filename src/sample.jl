# Sample from arbitrary arrays

function sample{T <: Real}(a::AbstractArray, probs::Vector{T})
    i = rand(Categorical(probs))
    return a[i]
end

function sample(a::AbstractArray)
    n = length(a)
    probs = ones(n) ./ n
    return sample(a, probs)
end
