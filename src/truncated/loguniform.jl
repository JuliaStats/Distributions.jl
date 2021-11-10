truncated(d::LogUniform, lo::T, hi::T) where {T<:Real} = truncated_LogUniform(d,lo,hi)
truncated(d::LogUniform, lo::Integer, hi::Integer) = truncated_LogUniform(d,lo,hi)

function truncated_LogUniform(d::LogUniform, lo, hi)
    a,b = params(d)
    LogUniform(max(a, lo), min(b, hi))
end
