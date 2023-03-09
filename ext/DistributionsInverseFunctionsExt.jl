module DistributionsInverseFunctionsExt

import InverseFunctions
using Distributions


for (fcdf, fqua) in (
    (:cdf, :quantile),
    (:ccdf, :cquantile),
)
    # unbounded distribution: can invert cdf at any point in [0..1]
    # bounded distribution: cannot invert cdf at 0 and 1
    finvcdf = Symbol(:_inv_, fcdf)
    @eval function $finvcdf(d, x)
        is_defined = zero(x) < x < one(x) || iszero(x) && !islowerbounded(d) || isone(x) && !isupperbounded(d)
        is_defined || error("inverse for $($fcdf)($d, x) is not defined at $x")
        return $fqua(d, x)
    end
end

for (fcdf, fqua) in (
    (:logcdf, :invlogcdf),
    (:logccdf, :invlogccdf),
)
    # unbounded distribution: can invert logcdf at any point in [-Inf..0]
    # bounded distribution: cannot invert logcdf at -Inf and 0
    finvcdf = Symbol(:_inv_, fcdf)
    @eval function $finvcdf(d, x)
        is_defined = typemin(typeof(x)) < x < zero(x) || typemin(typeof(x)) == x && !islowerbounded(d) || iszero(x) && !isupperbounded(d)
        is_defined || error("inverse for $($fcdf)($d, x) is not defined at $x")
        return $fqua(d, x)
    end
end

for (fcdf, fqua) in (
    (:cdf, :quantile),
    (:ccdf, :cquantile),
    (:logcdf, :invlogcdf),
    (:logccdf, :invlogccdf),
)
    finvqua = Symbol(:_inv_, fqua)
    finvcdf = Symbol(:_inv_, fcdf)

    # same as fcdf, but with the inverse being fqua directly
    # finvqua should only be used at the fqua inverse
    @eval $finvqua(d, x) = $fcdf(d, x)

    @eval InverseFunctions.inverse(f::Base.Fix1{typeof($fcdf), <:ContinuousUnivariateDistribution}) = Base.Fix1($finvcdf, f.x)
    @eval InverseFunctions.inverse(f::Base.Fix1{typeof($finvcdf), <:ContinuousUnivariateDistribution}) = Base.Fix1($fcdf, f.x)
    @eval InverseFunctions.inverse(f::Base.Fix1{typeof($fqua), <:ContinuousUnivariateDistribution}) = Base.Fix1($finvqua, f.x)
    @eval InverseFunctions.inverse(f::Base.Fix1{typeof($finvqua), <:ContinuousUnivariateDistribution}) = Base.Fix1($fqua, f.x)
end

end
