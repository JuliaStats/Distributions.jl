for (fcdf, fqua) in [
    (:cdf, :quantile),
    (:ccdf, :cquantile),
    (:logcdf, :invlogcdf),
    (:logccdf, :invlogccdf),
]
    @eval InverseFunctions.inverse(f::Base.Fix1{typeof($fcdf), <:ContinuousUnivariateDistribution}) = Base.Fix1($fqua, f.x)
    @eval InverseFunctions.inverse(f::Base.Fix1{typeof($fqua), <:ContinuousUnivariateDistribution}) = Base.Fix1($fcdf, f.x)
end
