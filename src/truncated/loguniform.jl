function truncated(d::LogUniform, lo::T, hi::T) where {T<:Real}
    a, b = params(d)
    return LogUniform(max(a, lo), min(b, hi))
end

# fix method ambiguity
truncated(d::LogUniform, lo::Integer, hi::Integer) = truncated(d, float(lo), float(hi))
