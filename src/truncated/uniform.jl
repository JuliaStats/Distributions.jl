#####
##### Shortcut for truncating uniform distributions.
#####

Truncated(d::Uniform, l::Float64, u::Float64) = Uniform(max(l, d.a), min(u, d.b))
