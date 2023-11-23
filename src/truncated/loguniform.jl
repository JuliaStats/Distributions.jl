truncated(d::LogUniform, lo::T, hi::T) where {T<:Real} = LogUniform(max(d.a, lo), min(d.b, hi))
truncated(d::LogUniform, lo::Real, ::Nothing) = LogUniform(max(d.a, lo), d.b)
truncated(d::LogUniform, ::Nothing, hi::Real) = LogUniform(d.a, min(d.b, hi))
