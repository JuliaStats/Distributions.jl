#####
##### Shortcut for truncating uniform distributions.
#####

truncated(d::Uniform, l::T, u::T) where {T <: Real} = Uniform(max(l, d.a), min(u, d.b))
