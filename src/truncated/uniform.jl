#####
##### Shortcut for truncating uniform distributions.
#####

Truncated(d::Uniform, l::Real, u::Real) = Uniform(promote(max(l, d.a), min(u, d.b))...)
