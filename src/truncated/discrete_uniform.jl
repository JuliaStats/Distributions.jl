#####
##### Shortcut for truncating discrete uniform distributions.
#####

function truncated(d::DiscreteUniform, l::T, u::T) where {T <: Real}
    a = round(max(l, d.a), RoundUp)
    b = round(min(u, d.b), RoundDown)
    return DiscreteUniform(a, b)
end
