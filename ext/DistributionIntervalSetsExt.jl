module DistributionIntervalSetsExt

using Distribution: Uniform, LogUniform, truncated, censored, RealInterval
using IntervalSets

Uniform(i::Interval) = Uniform(leftendpoint(i), rightendpoint(i))
LogUniform(i::Interval) = LogUniform(leftendpoint(i), rightendpoint(i))
truncated(d0, i::Interval) = truncated(d, leftendpoint(i), rightendpoint(i))
censored(d0, i::Interval) = censored(d, leftendpoint(i), rightendpoint(i))

Base.convert(::Type{T}, ri::RealInterval) where {T<:Interval} = T(minimum(ri), maximum(ri))

end
