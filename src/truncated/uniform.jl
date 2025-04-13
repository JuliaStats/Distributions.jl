#####
##### Shortcut for truncating uniform distributions.
#####

truncated(d::Uniform, l::T, u::T) where {T <: Real} = Uniform(max(l, d.a), min(u, d.b))
truncated(d::Uniform, l::Real, ::Nothing) = Uniform(max(l, d.a), d.b)
truncated(d::Uniform, ::Nothing, u::Real) = Uniform(d.a, min(u, d.b))
