truncated(d::LogUniform, lo::T, hi::T) where {T<:Real} = LogUniform(max(d.a, lo), min(d.b, hi))
