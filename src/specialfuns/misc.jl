
# the largest x such that exp(x) < Inf
realmaxexp{T<:FloatingPoint}(::Type{T}) = with_rounding(()->log(realmax(T)),RoundDown)
realmaxexp(::Type{BigFloat}) = with_bigfloat_rounding(()->log(prevfloat(inf(BigFloat))),RoundDown)

# the smallest x such that exp(x) > 0 (or at least not a subnormal number)
realminexp{T<:FloatingPoint}(::Type{T}) = with_rounding(()->log(realmin(T)),RoundUp)
realminexp(::Type{BigFloat}) = with_bigfloat_rounding(()->log(nextfloat(zero(BigFloat))),RoundUp)
