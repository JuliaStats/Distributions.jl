using Distributions
using Base.Test

# Mathematica code to generate this.

### entropy[dist_] :=
###  TimeConstrained[
###   Expectation[-Log[PDF[dist, x]], x \[Distributed] dist], 5,
###   None];(* Stop after a few seconds, as this computation is very very \
### slow. *)
### test[\[Mu]_, \[Sigma]_, \[Xi]_] :=
###  Module[{
###    dist = MaxStableDistribution[\[Mu], \[Sigma], \[Xi]], cdf,
###    q = {10, 25, 50, 75, 90}/100,
###    ent, max, min, mean, median, var, skew, kurt,
###    points, pointsCDF, pointsPDF, quantiles,
###    code, printv
###    },
###   cdf = CDF[dist]@x;
###
###   (* Compute reference results. *)
###   ent = entropy[dist];
###   min = If[
###     Norm[\[Xi]] < 1 E - 15 || \[Xi] <
###       0, -\[Infinity], \[Mu] - \[Sigma]/\[Xi]];
###   max = If[
###     Norm[\[Xi]] < 1 E - 15 || \[Xi] >
###       0, \[Infinity], \[Mu] - \[Sigma]/\[Xi]];
###   mean = Mean[dist]; (* Can be Indeterminate. *)
###   median = Median[dist];
###   var = Variance[dist]; (* Can be Indeterminate. *)
###   skew = Skewness[dist]; (* Can be Indeterminate. *)
###   kurt = Kurtosis[dist] - 3.0; (* Can be Indeterminate. \
###     Mathematica returns the kurtosis, \
###     Distributions.jl returns the excess kurtosis. *)
###
###   points =
###    Table[x /. FindRoot[cdf - cdfv, {x, 1}][[1]], {cdfv, .1, .9, .1}];
###   pointsCDF = CDF[dist, #] & /@ points;
###   pointsPDF = PDF[dist, #] & /@ points;
###
###   quantiles = Quantile[dist, #] & /@ q;
###
###   (* Output test code. *)
###   printv[v_] :=
###    If[v == Infinity, "Inf",
###     If[v == -Infinity, "-Inf", TextString[N[v]]], "Inf"]; (*
###   Handles Indeterminate through If's fourth argument. *)
###
###   code = "d = GeneralizedExtremeValue(" <> printv[\[Mu]] <> ", " <>
###     printv[\[Sigma]] <> ", " <> printv[\[Xi]] <> ")\n"
###     <> "@test_approx_eq location(d) " <> printv[\[Mu]] <> "\n"
###     <> "@test_approx_eq scale(d) " <> printv[\[Sigma]] <> "\n"
###     <> "@test_approx_eq shape(d) " <> printv[\[Xi]] <> "\n"
###     <> "@test_approx_eq_eps maximum(d) " <> printv[max] <> " 1e-5\n"
###     <> "@test_approx_eq_eps minimum(d) " <> printv[min] <> " 1e-5\n"
###     <> If[ent != None,
###      "@test_approx_eq_eps entropy(d) " <> printv[ent] <> " 1e-5\n",
###      "", ""]
###     <> "@test_approx_eq_eps mean(d) " <> printv[mean] <> " 1e-5\n"
###     <> "@test_approx_eq_eps median(d) " <> printv[median] <> " 1e-5\n"
###     <> "@test_approx_eq_eps var(d) " <> printv[var] <> " 1e-5\n"
###     <> "@test_approx_eq_eps skewness(d) " <> printv[skew] <> " 1e-5\n"
###     <> "@test_approx_eq_eps kurtosis(d) " <> printv[kurt] <> " 1e-5\n"
###     <> ((
###         "@test_approx_eq_eps pdf(d, " <> printv[points[[#]]] <> ") " <>
###           printv[pointsPDF[[#]]] <> " 1e-5\n"
###          <>
###          "@test_approx_eq_eps cdf(d, " <> printv[points[[#]]] <>
###          ") " <> printv[pointsCDF[[#]]] <> " 1e-5\n"
###          <>
###          "@test_approx_eq_eps logpdf(d, " <> printv[points[[#]]] <>
###          ") " <> printv[Log[pointsPDF[[#]]]] <> " 1e-5\n"
###          <>
###          "@test_approx_eq_eps logcdf(d, " <> printv[points[[#]]] <>
###          ") " <> printv[Log[pointsCDF[[#]]]] <> " 1e-5\n"
###          <>
###          "@test_approx_eq_eps ccdf(d, " <> printv[points[[#]]] <>
###          ") " <> printv[1 - pointsCDF[[#]]] <> " 1e-5\n"
###          <>
###          "@test_approx_eq_eps logccdf(d, " <> printv[points[[#]]] <>
###          ") " <> printv[Log[1 - pointsCDF[[#]]]] <> " 1e-5\n"
###         ) & /@ Table[i, {i, 1, Length@points}])
###     <> ((
###         "@test_approx_eq_eps quantile(d, " <> printv[q[[#]]] <> ") " <>
###           printv[quantiles[[#]]] <> " 1e-5\n"
###         ) & /@ Table[i, {i, 1, Length@quantiles}]);
###
###   Return[code];
###   ];

d = GeneralizedExtremeValue(1., 1., 1.)
@test GeneralizedExtremeValue(1., 1, 1) == d
@test GeneralizedExtremeValue(1, 1, 1) == d
@test typeof(convert(GeneralizedExtremeValue{Float32}, d)) == GeneralizedExtremeValue{Float32}
@test typeof(convert(GeneralizedExtremeValue{Float32}, 1, 1, 1)) == GeneralizedExtremeValue{Float32}

# test[1., 1., 1.]
d = GeneralizedExtremeValue(1., 1., 1.)
@test location(d) ≈ 1.
@test scale(d)    ≈ 1.
@test shape(d)    ≈ 1.
@test isapprox(maximum(d)           ,  Inf       , atol=1e-5)
@test isapprox(minimum(d)           ,  0.        , atol=1e-5)
@test isapprox(mean(d)              ,  Inf       , atol=1e-5)
@test isapprox(median(d)            ,  1.4427    , atol=1e-5)
@test isapprox(var(d)               ,  Inf       , atol=1e-5)
@test isapprox(skewness(d)          ,  Inf       , atol=1e-5)
@test isapprox(kurtosis(d)          ,  Inf       , atol=1e-5)
@test isapprox(pdf(d, 0.434294)     ,  0.53019   , atol=1e-5)
@test isapprox(cdf(d, 0.434294)     ,  0.1       , atol=1e-5)
@test isapprox(logpdf(d, 0.434294)  , -0.63452   , atol=1e-5)
@test isapprox(logcdf(d, 0.434294)  , -2.30259   , atol=1e-5)
@test isapprox(ccdf(d, 0.434294)    ,  0.9       , atol=1e-5)
@test isapprox(logccdf(d, 0.434294) , -0.105361  , atol=1e-5)
@test isapprox(pdf(d, 0.621335)     ,  0.518058  , atol=1e-5)
@test isapprox(cdf(d, 0.621335)     ,  0.2       , atol=1e-5)
@test isapprox(logpdf(d, 0.621335)  , -0.657668  , atol=1e-5)
@test isapprox(logcdf(d, 0.621335)  , -1.60944   , atol=1e-5)
@test isapprox(ccdf(d, 0.621335)    ,  0.8       , atol=1e-5)
@test isapprox(logccdf(d, 0.621335) , -0.223144  , atol=1e-5)
@test isapprox(pdf(d, 0.830584)     ,  0.434865  , atol=1e-5)
@test isapprox(cdf(d, 0.830584)     ,  0.3       , atol=1e-5)
@test isapprox(logpdf(d, 0.830584)  , -0.832719  , atol=1e-5)
@test isapprox(logcdf(d, 0.830584)  , -1.20397   , atol=1e-5)
@test isapprox(ccdf(d, 0.830584)    ,  0.7       , atol=1e-5)
@test isapprox(logccdf(d, 0.830584) , -0.356675  , atol=1e-5)
@test isapprox(pdf(d, 1.09136)      ,  0.335835  , atol=1e-5)
@test isapprox(cdf(d, 1.09136)      ,  0.4       , atol=1e-5)
@test isapprox(logpdf(d, 1.09136)   , -1.09113   , atol=1e-5)
@test isapprox(logcdf(d, 1.09136)   , -0.916291  , atol=1e-5)
@test isapprox(ccdf(d, 1.09136)     ,  0.6       , atol=1e-5)
@test isapprox(logccdf(d, 1.09136)  , -0.510826  , atol=1e-5)
@test isapprox(pdf(d, 1.4427)       ,  0.240227  , atol=1e-5)
@test isapprox(cdf(d, 1.4427)       ,  0.5       , atol=1e-5)
@test isapprox(logpdf(d, 1.4427)    , -1.42617   , atol=1e-5)
@test isapprox(logcdf(d, 1.4427)    , -0.693147  , atol=1e-5)
@test isapprox(ccdf(d, 1.4427)      ,  0.5       , atol=1e-5)
@test isapprox(logccdf(d, 1.4427)   , -0.693147  , atol=1e-5)
@test isapprox(pdf(d, 1.95762)      ,  0.156566  , atol=1e-5)
@test isapprox(cdf(d, 1.95762)      ,  0.6       , atol=1e-5)
@test isapprox(logpdf(d, 1.95762)   , -1.85428   , atol=1e-5)
@test isapprox(logcdf(d, 1.95762)   , -0.510826  , atol=1e-5)
@test isapprox(ccdf(d, 1.95762)     ,  0.4       , atol=1e-5)
@test isapprox(logccdf(d, 1.95762)  , -0.916291  , atol=1e-5)
@test isapprox(pdf(d, 2.80367)      ,  0.0890519 , atol=1e-5)
@test isapprox(cdf(d, 2.80367)      ,  0.7       , atol=1e-5)
@test isapprox(logpdf(d, 2.80367)   , -2.41854   , atol=1e-5)
@test isapprox(logcdf(d, 2.80367)   , -0.356675  , atol=1e-5)
@test isapprox(ccdf(d, 2.80367)     ,  0.3       , atol=1e-5)
@test isapprox(logccdf(d, 2.80367)  , -1.20397   , atol=1e-5)
@test isapprox(pdf(d, 4.48142)      ,  0.0398344 , atol=1e-5)
@test isapprox(cdf(d, 4.48142)      ,  0.8       , atol=1e-5)
@test isapprox(logpdf(d, 4.48142)   , -3.22302   , atol=1e-5)
@test isapprox(logcdf(d, 4.48142)   , -0.223144  , atol=1e-5)
@test isapprox(ccdf(d, 4.48142)     ,  0.2       , atol=1e-5)
@test isapprox(logccdf(d, 4.48142)  , -1.60944   , atol=1e-5)
@test isapprox(pdf(d, 9.49122)      ,  0.00999075, atol=1e-5)
@test isapprox(cdf(d, 9.49122)      ,  0.9       , atol=1e-5)
@test isapprox(logpdf(d, 9.49122)   , -4.6061    , atol=1e-5)
@test isapprox(logcdf(d, 9.49122)   , -0.105361  , atol=1e-5)
@test isapprox(ccdf(d, 9.49122)     ,  0.1       , atol=1e-5)
@test isapprox(logccdf(d, 9.49122)  , -2.30259   , atol=1e-5)
@test isapprox(quantile(d, 0.1)     ,  0.434294  , atol=1e-5)
@test isapprox(quantile(d, 0.25)    ,  0.721348  , atol=1e-5)
@test isapprox(quantile(d, 0.5)     ,  1.4427    , atol=1e-5)
@test isapprox(quantile(d,0.75)     ,  3.47606   , atol=1e-5)
@test isapprox(quantile(d, 0.9)     ,  9.49122   , atol=1e-5)

# test[0., 1., 0.]
d = GeneralizedExtremeValue(0., 1., 0.)
@test location(d) ≈ 0.
@test scale(d)    ≈ 1.
@test shape(d)    ≈ 0.
@test isapprox(maximum(d)            ,  Inf      , atol=1e-5)
@test isapprox(minimum(d)            , -Inf      , atol=1e-5)
@test isapprox(mean(d)               ,  0.577216 , atol=1e-5)
@test isapprox(median(d)             ,  0.366513 , atol=1e-5)
@test isapprox(var(d)                ,  1.64493  , atol=1e-5)
@test isapprox(skewness(d)           ,  1.13955  , atol=1e-5)
@test isapprox(kurtosis(d)           ,  2.4      , atol=1e-5)
@test isapprox(pdf(d, -0.834032)     ,  0.230259 , atol=1e-5)
@test isapprox(cdf(d, -0.834032)     ,  0.1      , atol=1e-5)
@test isapprox(logpdf(d, -0.834032)  , -1.46855  , atol=1e-5)
@test isapprox(logcdf(d, -0.834032)  , -2.30259  , atol=1e-5)
@test isapprox(ccdf(d, -0.834032)    ,  0.9      , atol=1e-5)
@test isapprox(logccdf(d, -0.834032) , -0.105361 , atol=1e-5)
@test isapprox(pdf(d, -0.475885)     ,  0.321888 , atol=1e-5)
@test isapprox(cdf(d, -0.475885)     ,  0.2      , atol=1e-5)
@test isapprox(logpdf(d, -0.475885)  , -1.13355  , atol=1e-5)
@test isapprox(logcdf(d, -0.475885)  , -1.60944  , atol=1e-5)
@test isapprox(ccdf(d, -0.475885)    ,  0.8      , atol=1e-5)
@test isapprox(logccdf(d, -0.475885) , -0.223144 , atol=1e-5)
@test isapprox(pdf(d, -0.185627)     ,  0.361192 , atol=1e-5)
@test isapprox(cdf(d, -0.185627)     ,  0.3      , atol=1e-5)
@test isapprox(logpdf(d, -0.185627)  , -1.01835  , atol=1e-5)
@test isapprox(logcdf(d, -0.185627)  , -1.20397  , atol=1e-5)
@test isapprox(ccdf(d, -0.185627)    ,  0.7      , atol=1e-5)
@test isapprox(logccdf(d, -0.185627) , -0.356675 , atol=1e-5)
@test isapprox(pdf(d, 0.0874216)     ,  0.366516 , atol=1e-5)
@test isapprox(cdf(d, 0.0874216)     ,  0.4      , atol=1e-5)
@test isapprox(logpdf(d, 0.0874216)  , -1.00371  , atol=1e-5)
@test isapprox(logcdf(d, 0.0874216)  , -0.916291 , atol=1e-5)
@test isapprox(ccdf(d, 0.0874216)    ,  0.6      , atol=1e-5)
@test isapprox(logccdf(d, 0.0874216) , -0.510826 , atol=1e-5)
@test isapprox(pdf(d, 0.366513)      ,  0.346574 , atol=1e-5)
@test isapprox(cdf(d, 0.366513)      ,  0.5      , atol=1e-5)
@test isapprox(logpdf(d, 0.366513)   , -1.05966  , atol=1e-5)
@test isapprox(logcdf(d, 0.366513)   , -0.693147 , atol=1e-5)
@test isapprox(ccdf(d, 0.366513)     ,  0.5      , atol=1e-5)
@test isapprox(logccdf(d, 0.366513)  , -0.693147 , atol=1e-5)
@test isapprox(pdf(d, 0.671727)      ,  0.306495 , atol=1e-5)
@test isapprox(cdf(d, 0.671727)      ,  0.6      , atol=1e-5)
@test isapprox(logpdf(d, 0.671727)   , -1.18255  , atol=1e-5)
@test isapprox(logcdf(d, 0.671727)   , -0.510826 , atol=1e-5)
@test isapprox(ccdf(d, 0.671727)     ,  0.4      , atol=1e-5)
@test isapprox(logccdf(d, 0.671727)  , -0.916291 , atol=1e-5)
@test isapprox(pdf(d, 1.03093)       ,  0.249672 , atol=1e-5)
@test isapprox(cdf(d, 1.03093)       ,  0.7      , atol=1e-5)
@test isapprox(logpdf(d, 1.03093)    , -1.38761  , atol=1e-5)
@test isapprox(logcdf(d, 1.03093)    , -0.356675 , atol=1e-5)
@test isapprox(ccdf(d, 1.03093)      ,  0.3      , atol=1e-5)
@test isapprox(logccdf(d, 1.03093)   , -1.20397  , atol=1e-5)
@test isapprox(pdf(d, 1.49994)       ,  0.178515 , atol=1e-5)
@test isapprox(cdf(d, 1.49994)       ,  0.8      , atol=1e-5)
@test isapprox(logpdf(d, 1.49994)    , -1.72308  , atol=1e-5)
@test isapprox(logcdf(d, 1.49994)    , -0.223144 , atol=1e-5)
@test isapprox(ccdf(d, 1.49994)      ,  0.2      , atol=1e-5)
@test isapprox(logccdf(d, 1.49994)   , -1.60944  , atol=1e-5)
@test isapprox(pdf(d, 2.25037)       ,  0.0948245, atol=1e-5)
@test isapprox(cdf(d, 2.25037)       ,  0.9      , atol=1e-5)
@test isapprox(logpdf(d, 2.25037)    , -2.35573  , atol=1e-5)
@test isapprox(logcdf(d, 2.25037)    , -0.105361 , atol=1e-5)
@test isapprox(ccdf(d, 2.25037)      ,  0.1      , atol=1e-5)
@test isapprox(logccdf(d, 2.25037)   , -2.30259  , atol=1e-5)
@test isapprox(quantile(d, 0.1)      , -0.834032 , atol=1e-5)
@test isapprox(quantile(d, 0.25)     , -0.326634 , atol=1e-5)
@test isapprox(quantile(d, 0.5)      ,  0.366513 , atol=1e-5)
@test isapprox(quantile(d, 0.75)     ,  1.2459   , atol=1e-5)
@test isapprox(quantile(d, 0.9)      ,  2.25037  , atol=1e-5)

# test[0., 1., 1.1]
d = GeneralizedExtremeValue(0., 1., 1.1)
@test location(d) ≈ 0.
@test scale(d)    ≈ 1.
@test shape(d)    ≈ 1.1
@test isapprox(maximum(d)           ,   Inf       , atol=1e-5)
@test isapprox(minimum(d)           ,  -0.909091  , atol=1e-5)
@test isapprox(mean(d)              ,   Inf       , atol=1e-5)
@test isapprox(median(d)            ,   0.451411  , atol=1e-5)
@test isapprox(var(d)               ,   Inf       , atol=1e-5)
@test isapprox(skewness(d)          ,   Inf       , atol=1e-5)
@test isapprox(kurtosis(d)          ,   Inf       , atol=1e-5)
@test isapprox(pdf(d, -2.43627)     ,   0.        , atol=1e-5)
@test isapprox(cdf(d, -2.43627)     ,   0.        , atol=1e-5)
@test isapprox(logpdf(d, -2.43627)  ,  -Inf       , atol=1e-5)
@test isapprox(logcdf(d, -2.43627)  ,  -Inf       , atol=1e-5)
@test isapprox(ccdf(d, -2.43627)    ,   1.        , atol=1e-5)
@test isapprox(logccdf(d, -2.43627) ,   0.        , atol=1e-5)
@test isapprox(pdf(d, -1.75017)     ,   0.        , atol=1e-5)
@test isapprox(cdf(d, -1.75017)     ,   0.        , atol=1e-5)
@test isapprox(logpdf(d, -1.75017)  ,  -Inf       , atol=1e-5)
@test isapprox(logcdf(d, -1.75017)  ,  -Inf       , atol=1e-5)
@test isapprox(ccdf(d, -1.75017)    ,   1.        , atol=1e-5)
@test isapprox(logccdf(d, -1.75017) ,   0.        , atol=1e-5)
@test isapprox(pdf(d, -1.06408)     ,   0.        , atol=1e-5)
@test isapprox(cdf(d, -1.06408)     ,   0.        , atol=1e-5)
@test isapprox(logpdf(d, -1.06408)  ,  -Inf       , atol=1e-5)
@test isapprox(logcdf(d, -1.06408)  ,  -Inf       , atol=1e-5)
@test isapprox(ccdf(d, -1.06408)    ,   1.        , atol=1e-5)
@test isapprox(logccdf(d, -1.06408) ,   0.        , atol=1e-5)
@test isapprox(pdf(d, 0.091763)     ,   0.332912  , atol=1e-5)
@test isapprox(cdf(d, 0.091763)     ,   0.4       , atol=1e-5)
@test isapprox(logpdf(d, 0.091763)  ,  -1.09988   , atol=1e-5)
@test isapprox(logcdf(d, 0.091763)  ,  -0.916291  , atol=1e-5)
@test isapprox(ccdf(d, 0.091763)    ,   0.6       , atol=1e-5)
@test isapprox(logccdf(d, 0.091763) ,  -0.510826  , atol=1e-5)
@test isapprox(pdf(d, 0.451411)     ,   0.231581  , atol=1e-5)
@test isapprox(cdf(d, 0.451411)     ,   0.5       , atol=1e-5)
@test isapprox(logpdf(d, 0.451411)  ,  -1.46282   , atol=1e-5)
@test isapprox(logcdf(d, 0.451411)  ,  -0.693147  , atol=1e-5)
@test isapprox(ccdf(d, 0.451411)    ,   0.5       , atol=1e-5)
@test isapprox(logccdf(d, 0.451411) ,  -0.693147  , atol=1e-5)
@test isapprox(pdf(d, 0.99421)      ,   0.146394  , atol=1e-5)
@test isapprox(cdf(d, 0.99421)      ,   0.6       , atol=1e-5)
@test isapprox(logpdf(d, 0.99421)   ,  -1.92145   , atol=1e-5)
@test isapprox(logcdf(d, 0.99421)   ,  -0.510826  , atol=1e-5)
@test isapprox(ccdf(d, 0.99421)     ,   0.4       , atol=1e-5)
@test isapprox(logccdf(d, 0.99421)  ,  -0.916291  , atol=1e-5)
@test isapprox(pdf(d, 1.91649)      ,   0.0803287 , atol=1e-5)
@test isapprox(cdf(d, 1.91649)      ,   0.7       , atol=1e-5)
@test isapprox(logpdf(d, 1.91649)   ,  -2.52163   , atol=1e-5)
@test isapprox(logcdf(d, 1.91649)   ,  -0.356675  , atol=1e-5)
@test isapprox(ccdf(d, 1.91649)     ,   0.3       , atol=1e-5)
@test isapprox(logccdf(d, 1.91649)  ,  -1.20397   , atol=1e-5)
@test isapprox(pdf(d, 3.82421)      ,   0.034286  , atol=1e-5)
@test isapprox(cdf(d, 3.82421)      ,   0.8       , atol=1e-5)
@test isapprox(logpdf(d, 3.82421)   ,  -3.37302   , atol=1e-5)
@test isapprox(logcdf(d, 3.82421)   ,  -0.223144  , atol=1e-5)
@test isapprox(ccdf(d, 3.82421)     ,   0.2       , atol=1e-5)
@test isapprox(logccdf(d, 3.82421)  ,  -1.60944   , atol=1e-5)
@test isapprox(pdf(d, 9.89683)      ,   0.00797749, atol=1e-5)
@test isapprox(cdf(d, 9.89683)      ,   0.9       , atol=1e-5)
@test isapprox(logpdf(d, 9.89683)   ,  -4.83113   , atol=1e-5)
@test isapprox(logcdf(d, 9.89683)   ,  -0.105361  , atol=1e-5)
@test isapprox(ccdf(d, 9.89683)     ,   0.1       , atol=1e-5)
@test isapprox(logccdf(d, 9.89683)  ,  -2.30259   , atol=1e-5)
@test isapprox(quantile(d, 0.1)     ,  -0.545871  , atol=1e-5)
@test isapprox(quantile(d, 0.25)    ,  -0.274394  , atol=1e-5)
@test isapprox(quantile(d, 0.5)     ,   0.451411  , atol=1e-5)
@test isapprox(quantile(d, 0.75)    ,   2.67025   , atol=1e-5)
@test isapprox(quantile(d, 0.9)     ,   9.89683   , atol=1e-5)

# test[0., 1., .6]
d = GeneralizedExtremeValue(0., 1., 0.6)
@test location(d) ≈ 0.
@test scale(d)    ≈ 1.
@test shape(d)    ≈ 0.6
@test isapprox(maximum(d)            ,  Inf       , atol=1e-5)
@test isapprox(minimum(d)            , -1.66667   , atol=1e-5)
@test isapprox(mean(d)               ,  2.03027   , atol=1e-5)
@test isapprox(median(d)             ,  0.409936  , atol=1e-5)
@test isapprox(var(d)                ,  Inf       , atol=1e-5)
@test isapprox(skewness(d)           ,  Inf       , atol=1e-5)
@test isapprox(kurtosis(d)           ,  Inf       , atol=1e-5)
@test isapprox(pdf(d, -1.94901)      ,  0.        , atol=1e-5)
@test isapprox(cdf(d, -1.94901)      ,  0.        , atol=1e-5)
@test isapprox(logpdf(d, -1.94901)   , -Inf       , atol=1e-5)
@test isapprox(logcdf(d, -1.94901)   , -Inf       , atol=1e-5)
@test isapprox(ccdf(d, -1.94901)     ,  1.        , atol=1e-5)
@test isapprox(logccdf(d, -1.94901)  ,  0.        , atol=1e-5)
@test isapprox(pdf(d, -0.413975)     ,  0.428261  , atol=1e-5)
@test isapprox(cdf(d, -0.413975)     ,  0.2       , atol=1e-5)
@test isapprox(logpdf(d, -0.413975)  , -0.848022  , atol=1e-5)
@test isapprox(logcdf(d, -0.413975)  , -1.60944   , atol=1e-5)
@test isapprox(ccdf(d, -0.413975)    ,  0.8       , atol=1e-5)
@test isapprox(logccdf(d, -0.413975) , -0.223144  , atol=1e-5)
@test isapprox(pdf(d, -0.175663)     ,  0.403746  , atol=1e-5)
@test isapprox(cdf(d, -0.175663)     ,  0.3       , atol=1e-5)
@test isapprox(logpdf(d, -0.175663)  , -0.90697   , atol=1e-5)
@test isapprox(logcdf(d, -0.175663)  , -1.20397   , atol=1e-5)
@test isapprox(ccdf(d, -0.175663)    ,  0.7       , atol=1e-5)
@test isapprox(logccdf(d, -0.175663) , -0.356675  , atol=1e-5)
@test isapprox(pdf(d, 0.0897549)     ,  0.347787  , atol=1e-5)
@test isapprox(cdf(d, 0.0897549)     ,  0.4       , atol=1e-5)
@test isapprox(logpdf(d, 0.0897549)  , -1.05617   , atol=1e-5)
@test isapprox(logcdf(d, 0.0897549)  , -0.916291  , atol=1e-5)
@test isapprox(ccdf(d, 0.0897549)    ,  0.6       , atol=1e-5)
@test isapprox(logccdf(d, 0.0897549) , -0.510826  , atol=1e-5)
@test isapprox(pdf(d, 0.409936)      ,  0.278157  , atol=1e-5)
@test isapprox(cdf(d, 0.409936)      ,  0.5       , atol=1e-5)
@test isapprox(logpdf(d, 0.409936)   , -1.27957   , atol=1e-5)
@test isapprox(logcdf(d, 0.409936)   , -0.693147  , atol=1e-5)
@test isapprox(ccdf(d, 0.409936)     ,  0.5       , atol=1e-5)
@test isapprox(logccdf(d, 0.409936)  , -0.693147  , atol=1e-5)
@test isapprox(pdf(d, 0.827268)      ,  0.204827  , atol=1e-5)
@test isapprox(cdf(d, 0.827268)      ,  0.6       , atol=1e-5)
@test isapprox(logpdf(d, 0.827268)   , -1.58559   , atol=1e-5)
@test isapprox(logcdf(d, 0.827268)   , -0.510826  , atol=1e-5)
@test isapprox(ccdf(d, 0.827268)     ,  0.4       , atol=1e-5)
@test isapprox(logccdf(d, 0.827268)  , -0.916291  , atol=1e-5)
@test isapprox(pdf(d, 1.42708)       ,  0.134504  , atol=1e-5)
@test isapprox(cdf(d, 1.42708)       ,  0.7       , atol=1e-5)
@test isapprox(logpdf(d, 1.42708)    , -2.00616   , atol=1e-5)
@test isapprox(logcdf(d, 1.42708)    , -0.356675  , atol=1e-5)
@test isapprox(ccdf(d, 1.42708)      ,  0.3       , atol=1e-5)
@test isapprox(logccdf(d, 1.42708)   , -1.20397   , atol=1e-5)
@test isapprox(pdf(d, 2.43252)       ,  0.0725813 , atol=1e-5)
@test isapprox(cdf(d, 2.43252)       ,  0.8       , atol=1e-5)
@test isapprox(logpdf(d, 2.43252)    , -2.62305   , atol=1e-5)
@test isapprox(logcdf(d, 2.43252)    , -0.223144  , atol=1e-5)
@test isapprox(ccdf(d, 2.43252)      ,  0.2       , atol=1e-5)
@test isapprox(logccdf(d, 2.43252)   , -1.60944   , atol=1e-5)
@test isapprox(pdf(d, 4.76379)       ,  0.0245769 , atol=1e-5)
@test isapprox(cdf(d, 4.76379)       ,  0.9       , atol=1e-5)
@test isapprox(logpdf(d, 4.76379)    , -3.70595   , atol=1e-5)
@test isapprox(logcdf(d, 4.76379)    , -0.105361  , atol=1e-5)
@test isapprox(ccdf(d, 4.76379)      ,  0.1       , atol=1e-5)
@test isapprox(logccdf(d, 4.76379)   , -2.30259   , atol=1e-5)
@test isapprox(quantile(d, 0.1)      , -0.656206  , atol=1e-5)
@test isapprox(quantile(d, 0.25)     , -0.29662   , atol=1e-5)
@test isapprox(quantile(d, 0.5)      ,  0.409936  , atol=1e-5)
@test isapprox(quantile(d, 0.75)     ,  1.853     , atol=1e-5)
@test isapprox(quantile(d, 0.9)      ,  4.76379   , atol=1e-5)

# test[0., 1., .3]
d = GeneralizedExtremeValue(0., 1., 0.3)
@test location(d) ≈ 0.
@test scale(d)    ≈ 1.
@test shape(d)    ≈ 0.3
@test isapprox(maximum(d)            ,  Inf      , atol=1e-5)
@test isapprox(minimum(d)            , -3.33333  , atol=1e-5)
@test isapprox(mean(d)               ,  0.993518 , atol=1e-5)
@test isapprox(median(d)             ,  0.387422 , atol=1e-5)
@test isapprox(var(d)                ,  5.92458  , atol=1e-5)
@test isapprox(skewness(d)           , 13.4836   , atol=5e-5) # Changed precision!
@test isapprox(kurtosis(d)           ,  Inf      , atol=1e-5)
@test isapprox(pdf(d, -0.737875)     ,  0.29572  , atol=1e-5)
@test isapprox(cdf(d, -0.737875)     ,  0.1      , atol=1e-5)
@test isapprox(logpdf(d, -0.737875)  , -1.21834  , atol=1e-5)
@test isapprox(logcdf(d, -0.737875)  , -2.30259  , atol=1e-5)
@test isapprox(ccdf(d, -0.737875)    ,  0.9      , atol=1e-5)
@test isapprox(logccdf(d, -0.737875) , -0.105361 , atol=1e-5)
@test isapprox(pdf(d, -0.443476)     ,  0.371284 , atol=1e-5)
@test isapprox(cdf(d, -0.443476)     ,  0.2      , atol=1e-5)
@test isapprox(logpdf(d, -0.443476)  , -0.990787 , atol=1e-5)
@test isapprox(logcdf(d, -0.443476)  , -1.60944  , atol=1e-5)
@test isapprox(ccdf(d, -0.443476)    ,  0.8      , atol=1e-5)
@test isapprox(logccdf(d, -0.443476) , -0.223144 , atol=1e-5)
@test isapprox(pdf(d, -0.180553)     ,  0.381877 , atol=1e-5)
@test isapprox(cdf(d, -0.180553)     ,  0.3      , atol=1e-5)
@test isapprox(logpdf(d, -0.180553)  , -0.962658 , atol=1e-5)
@test isapprox(logcdf(d, -0.180553)  , -1.20397  , atol=1e-5)
@test isapprox(ccdf(d, -0.180553)    ,  0.7      , atol=1e-5)
@test isapprox(logccdf(d, -0.180553) , -0.356675 , atol=1e-5)
@test isapprox(pdf(d, 0.088578)      ,  0.357029 , atol=1e-5)
@test isapprox(cdf(d, 0.088578)      ,  0.4      , atol=1e-5)
@test isapprox(logpdf(d, 0.088578)   , -1.02994  , atol=1e-5)
@test isapprox(logcdf(d, 0.088578)   , -0.916291 , atol=1e-5)
@test isapprox(ccdf(d, 0.088578)     ,  0.6      , atol=1e-5)
@test isapprox(logccdf(d, 0.088578)  , -0.510826 , atol=1e-5)
@test isapprox(pdf(d, 0.387422)      ,  0.310487 , atol=1e-5)
@test isapprox(cdf(d, 0.387422)      ,  0.5      , atol=1e-5)
@test isapprox(logpdf(d, 0.387422)   , -1.16961  , atol=1e-5)
@test isapprox(logcdf(d, 0.387422)   , -0.693147 , atol=1e-5)
@test isapprox(ccdf(d, 0.387422)     ,  0.5      , atol=1e-5)
@test isapprox(logccdf(d, 0.387422)  , -0.693147 , atol=1e-5)
@test isapprox(pdf(d, 0.744195)      ,  0.250557 , atol=1e-5)
@test isapprox(cdf(d, 0.744195)      ,  0.6      , atol=1e-5)
@test isapprox(logpdf(d, 0.744195)   , -1.38407  , atol=1e-5)
@test isapprox(logcdf(d, 0.744195)   , -0.510826 , atol=1e-5)
@test isapprox(ccdf(d, 0.744195)     ,  0.4      , atol=1e-5)
@test isapprox(logccdf(d, 0.744195)  , -0.916291 , atol=1e-5)
@test isapprox(pdf(d, 1.20814)       ,  0.183254 , atol=1e-5)
@test isapprox(cdf(d, 1.20814)       ,  0.7      , atol=1e-5)
@test isapprox(logpdf(d, 1.20814)    , -1.69688  , atol=1e-5)
@test isapprox(logcdf(d, 1.20814)    , -0.356675 , atol=1e-5)
@test isapprox(ccdf(d, 1.20814)      ,  0.3      , atol=1e-5)
@test isapprox(logccdf(d, 1.20814)   , -1.20397  , atol=1e-5)
@test isapprox(pdf(d, 1.89428)       ,  0.113828 , atol=1e-5)
@test isapprox(cdf(d, 1.89428)       ,  0.8      , atol=1e-5)
@test isapprox(logpdf(d, 1.89428)    , -2.17307  , atol=1e-5)
@test isapprox(logcdf(d, 1.89428)    , -0.223144 , atol=1e-5)
@test isapprox(ccdf(d, 1.89428)      ,  0.2      , atol=1e-5)
@test isapprox(logccdf(d, 1.89428)   , -1.60944  , atol=1e-5)
@test isapprox(pdf(d, 3.21416)       ,  0.0482752, atol=1e-5)
@test isapprox(cdf(d, 3.21416)       ,  0.9      , atol=1e-5)
@test isapprox(logpdf(d, 3.21416)    , -3.03084  , atol=1e-5)
@test isapprox(logcdf(d, 3.21416)    , -0.105361 , atol=1e-5)
@test isapprox(ccdf(d, 3.21416)      ,  0.1      , atol=1e-5)
@test isapprox(logccdf(d, 3.21416)   , -2.30259  , atol=1e-5)
@test isapprox(quantile(d, 0.1)      , -0.737875 , atol=1e-5)
@test isapprox(quantile(d, 0.25)     , -0.311141 , atol=1e-5)
@test isapprox(quantile(d, 0.5)      ,  0.387422 , atol=1e-5)
@test isapprox(quantile(d, 0.75)     ,  1.51068  , atol=1e-5)
@test isapprox(quantile(d, 0.9)      ,  3.21416  , atol=1e-5)

# test[1., 1., -1.]
d = GeneralizedExtremeValue(1., 1., -1.)
@test location(d) ≈ 1.
@test scale(d)    ≈ 1.
@test shape(d)    ≈ -1.
@test isapprox(maximum(d)            ,  2.      , atol=1e-5)
@test isapprox(minimum(d)            , -Inf     , atol=1e-5)
@test isapprox(mean(d)               ,  1.      , atol=1e-5)
@test isapprox(median(d)             ,  1.30685 , atol=1e-5)
@test isapprox(var(d)                ,  1.      , atol=1e-5)
@test isapprox(skewness(d)           , -2.      , atol=1e-5)
@test isapprox(kurtosis(d)           ,  6.      , atol=1e-5)
@test isapprox(pdf(d, -0.302585)     ,  0.1     , atol=1e-5)
@test isapprox(cdf(d, -0.302585)     ,  0.1     , atol=1e-5)
@test isapprox(logpdf(d, -0.302585)  , -2.30259 , atol=1e-5)
@test isapprox(logcdf(d, -0.302585)  , -2.30259 , atol=1e-5)
@test isapprox(ccdf(d, -0.302585)    ,  0.9     , atol=1e-5)
@test isapprox(logccdf(d, -0.302585) , -0.105361, atol=1e-5)
@test isapprox(pdf(d, 0.390562)      ,  0.2     , atol=1e-5)
@test isapprox(cdf(d, 0.390562)      ,  0.2     , atol=1e-5)
@test isapprox(logpdf(d, 0.390562)   , -1.60944 , atol=1e-5)
@test isapprox(logcdf(d, 0.390562)   , -1.60944 , atol=1e-5)
@test isapprox(ccdf(d, 0.390562)     ,  0.8     , atol=1e-5)
@test isapprox(logccdf(d, 0.390562)  , -0.223144, atol=1e-5)
@test isapprox(pdf(d, 0.796027)      ,  0.3     , atol=1e-5)
@test isapprox(cdf(d, 0.796027)      ,  0.3     , atol=1e-5)
@test isapprox(logpdf(d, 0.796027)   , -1.20397 , atol=1e-5)
@test isapprox(logcdf(d, 0.796027)   , -1.20397 , atol=1e-5)
@test isapprox(ccdf(d, 0.796027)     ,  0.7     , atol=1e-5)
@test isapprox(logccdf(d, 0.796027)  , -0.356675, atol=1e-5)
@test isapprox(pdf(d, 1.08371)       ,  0.4     , atol=1e-5)
@test isapprox(cdf(d, 1.08371)       ,  0.4     , atol=1e-5)
@test isapprox(logpdf(d, 1.08371)    , -0.916291, atol=1e-5)
@test isapprox(logcdf(d, 1.08371)    , -0.916291, atol=1e-5)
@test isapprox(ccdf(d, 1.08371)      ,  0.6     , atol=1e-5)
@test isapprox(logccdf(d, 1.08371)   , -0.510826, atol=1e-5)
@test isapprox(pdf(d, 1.30685)       ,  0.5     , atol=1e-5)
@test isapprox(cdf(d, 1.30685)       ,  0.5     , atol=1e-5)
@test isapprox(logpdf(d, 1.30685)    , -0.693147, atol=1e-5)
@test isapprox(logcdf(d, 1.30685)    , -0.693147, atol=1e-5)
@test isapprox(ccdf(d, 1.30685)      ,  0.5     , atol=1e-5)
@test isapprox(logccdf(d, 1.30685)   , -0.693147, atol=1e-5)
@test isapprox(pdf(d, 1.48917)       ,  0.6     , atol=1e-5)
@test isapprox(cdf(d, 1.48917)       ,  0.6     , atol=1e-5)
@test isapprox(logpdf(d, 1.48917)    , -0.510826, atol=1e-5)
@test isapprox(logcdf(d, 1.48917)    , -0.510826, atol=1e-5)
@test isapprox(ccdf(d, 1.48917)      ,  0.4     , atol=1e-5)
@test isapprox(logccdf(d, 1.48917)   , -0.916291, atol=1e-5)
@test isapprox(pdf(d, 1.64333)       ,  0.7     , atol=1e-5)
@test isapprox(cdf(d, 1.64333)       ,  0.7     , atol=1e-5)
@test isapprox(logpdf(d, 1.64333)    , -0.356675, atol=1e-5)
@test isapprox(logcdf(d, 1.64333)    , -0.356675, atol=1e-5)
@test isapprox(ccdf(d, 1.64333)      ,  0.3     , atol=1e-5)
@test isapprox(logccdf(d, 1.64333)   , -1.20397 , atol=5e-5) # Changed precision!
@test isapprox(pdf(d, 2.17463)       ,  0.      , atol=1e-5)
@test isapprox(cdf(d, 2.17463)       ,  1.      , atol=1e-5)
@test isapprox(logpdf(d, 2.17463)    , -Inf     , atol=1e-5)
@test isapprox(logcdf(d, 2.17463)    ,  0.      , atol=1e-5)
@test isapprox(ccdf(d, 2.17463)      ,  0.      , atol=1e-5)
@test isapprox(logccdf(d, 2.17463)   , -Inf     , atol=1e-5)
@test isapprox(pdf(d, 2.44645)       ,  0.      , atol=1e-5)
@test isapprox(cdf(d, 2.44645)       ,  1.      , atol=1e-5)
@test isapprox(logpdf(d, 2.44645)    , -Inf     , atol=1e-5)
@test isapprox(logcdf(d, 2.44645)    ,  0.      , atol=1e-5)
@test isapprox(ccdf(d, 2.44645)      ,  0.      , atol=1e-5)
@test isapprox(logccdf(d, 2.44645)   , -Inf     , atol=1e-5)
@test isapprox(quantile(d, 0.1)      , -0.302585, atol=1e-5)
@test isapprox(quantile(d, 0.25)     ,  0.613706, atol=1e-5)
@test isapprox(quantile(d, 0.5)      ,  1.30685 , atol=1e-5)
@test isapprox(quantile(d, 0.75)     ,  1.71232 , atol=1e-5)
@test isapprox(quantile(d, 0.9)      ,  1.89464 , atol=1e-5)

# test[-1, 0.5, 0.6]
d = GeneralizedExtremeValue(-1., 0.5, 0.6)
@test location(d) ≈ -1.
@test scale(d)    ≈  0.5
@test shape(d)    ≈  0.6
@test isapprox(maximum(d)            ,  Inf      , atol=1e-5)
@test isapprox(minimum(d)            , -1.83333  , atol=1e-5)
@test isapprox(mean(d)               ,  0.015133 , atol=1e-5)
@test isapprox(median(d)             , -0.795032 , atol=1e-5)
@test isapprox(var(d)                ,  Inf      , atol=1e-5)
@test isapprox(skewness(d)           ,  Inf      , atol=1e-5)
@test isapprox(kurtosis(d)           ,  Inf      , atol=1e-5)
@test isapprox(pdf(d, -10.5807)      ,  0.       , atol=1e-5)
@test isapprox(cdf(d, -10.5807)      ,  0.       , atol=1e-5)
@test isapprox(logpdf(d, -10.5807)   , -Inf      , atol=1e-5)
@test isapprox(logcdf(d, -10.5807)   , -Inf      , atol=1e-5)
@test isapprox(ccdf(d, -10.5807)     ,  1.       , atol=1e-5)
@test isapprox(logccdf(d, -10.5807)  ,  0.       , atol=1e-5)
@test isapprox(pdf(d, -9.09221)      ,  0.       , atol=1e-5)
@test isapprox(cdf(d, -9.09221)      ,  0.       , atol=1e-5)
@test isapprox(logpdf(d, -9.09221)   , -Inf      , atol=1e-5)
@test isapprox(logcdf(d, -9.09221)   , -Inf      , atol=1e-5)
@test isapprox(ccdf(d, -9.09221)     ,  1.       , atol=1e-5)
@test isapprox(logccdf(d, -9.09221)  ,  0.       , atol=1e-5)
@test isapprox(pdf(d, -7.60374)      ,  0.       , atol=1e-5)
@test isapprox(cdf(d, -7.60374)      ,  0.       , atol=1e-5)
@test isapprox(logpdf(d, -7.60374)   , -Inf      , atol=1e-5)
@test isapprox(logcdf(d, -7.60374)   , -Inf      , atol=1e-5)
@test isapprox(ccdf(d, -7.60374)     ,  1.       , atol=1e-5)
@test isapprox(logccdf(d, -7.60374)  ,  0.       , atol=1e-5)
@test isapprox(pdf(d, -6.11528)      ,  0.       , atol=1e-5)
@test isapprox(cdf(d, -6.11528)      ,  0.       , atol=1e-5)
@test isapprox(logpdf(d, -6.11528)   , -Inf      , atol=1e-5)
@test isapprox(logcdf(d, -6.11528)   , -Inf      , atol=1e-5)
@test isapprox(ccdf(d, -6.11528)     ,  1.       , atol=1e-5)
@test isapprox(logccdf(d, -6.11528)  ,  0.       , atol=1e-5)
@test isapprox(pdf(d, -0.795032)     ,  0.556315 , atol=1e-5)
@test isapprox(cdf(d, -0.795032)     ,  0.5      , atol=1e-5)
@test isapprox(logpdf(d, -0.795032)  , -0.586421 , atol=1e-5)
@test isapprox(logcdf(d, -0.795032)  , -0.693147 , atol=1e-5)
@test isapprox(ccdf(d, -0.795032)    ,  0.5      , atol=1e-5)
@test isapprox(logccdf(d, -0.795032) , -0.693147 , atol=1e-5)
@test isapprox(pdf(d, -0.586366)     ,  0.409654 , atol=1e-5)
@test isapprox(cdf(d, -0.586366)     ,  0.6      , atol=1e-5)
@test isapprox(logpdf(d, -0.586366)  , -0.892442 , atol=1e-5)
@test isapprox(logcdf(d, -0.586366)  , -0.510826 , atol=1e-5)
@test isapprox(ccdf(d, -0.586366)    ,  0.4      , atol=1e-5)
@test isapprox(logccdf(d, -0.586366) , -0.916291 , atol=1e-5)
@test isapprox(pdf(d, -0.286458)     ,  0.269007 , atol=1e-5)
@test isapprox(cdf(d, -0.286458)     ,  0.7      , atol=1e-5)
@test isapprox(logpdf(d, -0.286458)  , -1.31302  , atol=1e-5)
@test isapprox(logcdf(d, -0.286458)  , -0.356675 , atol=1e-5)
@test isapprox(ccdf(d, -0.286458)    ,  0.3      , atol=1e-5)
@test isapprox(logccdf(d, -0.286458) , -1.20397  , atol=1e-5)
@test isapprox(pdf(d, 0.216262)      ,  0.145163 , atol=1e-5)
@test isapprox(cdf(d, 0.216262)      ,  0.8      , atol=1e-5)
@test isapprox(logpdf(d, 0.216262)   , -1.9299   , atol=1e-5)
@test isapprox(logcdf(d, 0.216262)   , -0.223144 , atol=1e-5)
@test isapprox(ccdf(d, 0.216262)     ,  0.2      , atol=1e-5)
@test isapprox(logccdf(d, 0.216262)  , -1.60944  , atol=1e-5)
@test isapprox(pdf(d, 1.3819)        ,  0.0491538, atol=1e-5)
@test isapprox(cdf(d, 1.3819)        ,  0.9      , atol=1e-5)
@test isapprox(logpdf(d, 1.3819)     , -3.0128   , atol=1e-5)
@test isapprox(logcdf(d, 1.3819)     , -0.105361 , atol=1e-5)
@test isapprox(ccdf(d, 1.3819)       ,  0.1      , atol=1e-5)
@test isapprox(logccdf(d, 1.3819)    , -2.30259  , atol=1e-5)
@test isapprox(quantile(d, 0.1)      , -1.3281   , atol=1e-5)
@test isapprox(quantile(d, 0.25)     , -1.14831  , atol=1e-5)
@test isapprox(quantile(d, 0.5)      , -0.795032 , atol=1e-5)
@test isapprox(quantile(d, 0.75)     , -0.0735019, atol=1e-5)
@test isapprox(quantile(d, 0.9)      ,  1.3819   , atol=1e-5)
