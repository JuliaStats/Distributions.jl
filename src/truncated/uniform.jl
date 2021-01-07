#####
##### Shortcut for truncating uniform distributions.
#####

truncated(d::Uniform, l::T, u::T) where {T <: Real} = Uniform(promote(max(l, d.a), min(u, d.b))...)

truncated(d::Uniform, l::Integer, u::Integer) = truncated(d, float(l), float(u))
