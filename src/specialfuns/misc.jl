
# the largest x such that exp(x) < Inf
realmaxexp{T<:FloatingPoint}(::Type{T}) = prevfloat(log(realmax(T)))

# the smallest x such that exp(x) > 0 (or at least not a subnormal number)
realminexp{T<:FloatingPoint}(::Type{T}) = nextfloat(log(realmin(T)))
