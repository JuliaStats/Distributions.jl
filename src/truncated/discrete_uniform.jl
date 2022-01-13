#####
##### Shortcut for truncating discrete uniform distributions.
#####

function truncated(d::DiscreteUniform, l::T, u::T) where {T <: Real}
    a = ceil(Int, max(l, d.a))
    b = floor(Int, min(u, d.b))
    return DiscreteUniform(a, b)
end
