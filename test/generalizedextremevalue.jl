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
@test_approx_eq location(d) 1.
@test_approx_eq scale(d) 1.
@test_approx_eq shape(d) 1.
@test_approx_eq_eps maximum(d) Inf 1e-5
@test_approx_eq_eps minimum(d) 0. 1e-5
@test_approx_eq_eps mean(d) Inf 1e-5
@test_approx_eq_eps median(d) 1.4427 1e-5
@test_approx_eq_eps var(d) Inf 1e-5
@test_approx_eq_eps skewness(d) Inf 1e-5
@test_approx_eq_eps kurtosis(d) Inf 1e-5
@test_approx_eq_eps pdf(d, 0.434294) 0.53019 1e-5
@test_approx_eq_eps cdf(d, 0.434294) 0.1 1e-5
@test_approx_eq_eps logpdf(d, 0.434294) -0.63452 1e-5
@test_approx_eq_eps logcdf(d, 0.434294) -2.30259 1e-5
@test_approx_eq_eps ccdf(d, 0.434294) 0.9 1e-5
@test_approx_eq_eps logccdf(d, 0.434294) -0.105361 1e-5
@test_approx_eq_eps pdf(d, 0.621335) 0.518058 1e-5
@test_approx_eq_eps cdf(d, 0.621335) 0.2 1e-5
@test_approx_eq_eps logpdf(d, 0.621335) -0.657668 1e-5
@test_approx_eq_eps logcdf(d, 0.621335) -1.60944 1e-5
@test_approx_eq_eps ccdf(d, 0.621335) 0.8 1e-5
@test_approx_eq_eps logccdf(d, 0.621335) -0.223144 1e-5
@test_approx_eq_eps pdf(d, 0.830584) 0.434865 1e-5
@test_approx_eq_eps cdf(d, 0.830584) 0.3 1e-5
@test_approx_eq_eps logpdf(d, 0.830584) -0.832719 1e-5
@test_approx_eq_eps logcdf(d, 0.830584) -1.20397 1e-5
@test_approx_eq_eps ccdf(d, 0.830584) 0.7 1e-5
@test_approx_eq_eps logccdf(d, 0.830584) -0.356675 1e-5
@test_approx_eq_eps pdf(d, 1.09136) 0.335835 1e-5
@test_approx_eq_eps cdf(d, 1.09136) 0.4 1e-5
@test_approx_eq_eps logpdf(d, 1.09136) -1.09113 1e-5
@test_approx_eq_eps logcdf(d, 1.09136) -0.916291 1e-5
@test_approx_eq_eps ccdf(d, 1.09136) 0.6 1e-5
@test_approx_eq_eps logccdf(d, 1.09136) -0.510826 1e-5
@test_approx_eq_eps pdf(d, 1.4427) 0.240227 1e-5
@test_approx_eq_eps cdf(d, 1.4427) 0.5 1e-5
@test_approx_eq_eps logpdf(d, 1.4427) -1.42617 1e-5
@test_approx_eq_eps logcdf(d, 1.4427) -0.693147 1e-5
@test_approx_eq_eps ccdf(d, 1.4427) 0.5 1e-5
@test_approx_eq_eps logccdf(d, 1.4427) -0.693147 1e-5
@test_approx_eq_eps pdf(d, 1.95762) 0.156566 1e-5
@test_approx_eq_eps cdf(d, 1.95762) 0.6 1e-5
@test_approx_eq_eps logpdf(d, 1.95762) -1.85428 1e-5
@test_approx_eq_eps logcdf(d, 1.95762) -0.510826 1e-5
@test_approx_eq_eps ccdf(d, 1.95762) 0.4 1e-5
@test_approx_eq_eps logccdf(d, 1.95762) -0.916291 1e-5
@test_approx_eq_eps pdf(d, 2.80367) 0.0890519 1e-5
@test_approx_eq_eps cdf(d, 2.80367) 0.7 1e-5
@test_approx_eq_eps logpdf(d, 2.80367) -2.41854 1e-5
@test_approx_eq_eps logcdf(d, 2.80367) -0.356675 1e-5
@test_approx_eq_eps ccdf(d, 2.80367) 0.3 1e-5
@test_approx_eq_eps logccdf(d, 2.80367) -1.20397 1e-5
@test_approx_eq_eps pdf(d, 4.48142) 0.0398344 1e-5
@test_approx_eq_eps cdf(d, 4.48142) 0.8 1e-5
@test_approx_eq_eps logpdf(d, 4.48142) -3.22302 1e-5
@test_approx_eq_eps logcdf(d, 4.48142) -0.223144 1e-5
@test_approx_eq_eps ccdf(d, 4.48142) 0.2 1e-5
@test_approx_eq_eps logccdf(d, 4.48142) -1.60944 1e-5
@test_approx_eq_eps pdf(d, 9.49122) 0.00999075 1e-5
@test_approx_eq_eps cdf(d, 9.49122) 0.9 1e-5
@test_approx_eq_eps logpdf(d, 9.49122) -4.6061 1e-5
@test_approx_eq_eps logcdf(d, 9.49122) -0.105361 1e-5
@test_approx_eq_eps ccdf(d, 9.49122) 0.1 1e-5
@test_approx_eq_eps logccdf(d, 9.49122) -2.30259 1e-5
@test_approx_eq_eps quantile(d, 0.1) 0.434294 1e-5
@test_approx_eq_eps quantile(d, 0.25) 0.721348 1e-5
@test_approx_eq_eps quantile(d, 0.5) 1.4427 1e-5
@test_approx_eq_eps quantile(d, 0.75) 3.47606 1e-5
@test_approx_eq_eps quantile(d, 0.9) 9.49122 1e-5

# test[0., 1., 0.]
d = GeneralizedExtremeValue(0., 1., 0.)
@test_approx_eq location(d) 0.
@test_approx_eq scale(d) 1.
@test_approx_eq shape(d) 0.
@test_approx_eq_eps maximum(d) Inf 1e-5
@test_approx_eq_eps minimum(d) -Inf 1e-5
@test_approx_eq_eps mean(d) 0.577216 1e-5
@test_approx_eq_eps median(d) 0.366513 1e-5
@test_approx_eq_eps var(d) 1.64493 1e-5
@test_approx_eq_eps skewness(d) 1.13955 1e-5
@test_approx_eq_eps kurtosis(d) 2.4 1e-5
@test_approx_eq_eps pdf(d, -0.834032) 0.230259 1e-5
@test_approx_eq_eps cdf(d, -0.834032) 0.1 1e-5
@test_approx_eq_eps logpdf(d, -0.834032) -1.46855 1e-5
@test_approx_eq_eps logcdf(d, -0.834032) -2.30259 1e-5
@test_approx_eq_eps ccdf(d, -0.834032) 0.9 1e-5
@test_approx_eq_eps logccdf(d, -0.834032) -0.105361 1e-5
@test_approx_eq_eps pdf(d, -0.475885) 0.321888 1e-5
@test_approx_eq_eps cdf(d, -0.475885) 0.2 1e-5
@test_approx_eq_eps logpdf(d, -0.475885) -1.13355 1e-5
@test_approx_eq_eps logcdf(d, -0.475885) -1.60944 1e-5
@test_approx_eq_eps ccdf(d, -0.475885) 0.8 1e-5
@test_approx_eq_eps logccdf(d, -0.475885) -0.223144 1e-5
@test_approx_eq_eps pdf(d, -0.185627) 0.361192 1e-5
@test_approx_eq_eps cdf(d, -0.185627) 0.3 1e-5
@test_approx_eq_eps logpdf(d, -0.185627) -1.01835 1e-5
@test_approx_eq_eps logcdf(d, -0.185627) -1.20397 1e-5
@test_approx_eq_eps ccdf(d, -0.185627) 0.7 1e-5
@test_approx_eq_eps logccdf(d, -0.185627) -0.356675 1e-5
@test_approx_eq_eps pdf(d, 0.0874216) 0.366516 1e-5
@test_approx_eq_eps cdf(d, 0.0874216) 0.4 1e-5
@test_approx_eq_eps logpdf(d, 0.0874216) -1.00371 1e-5
@test_approx_eq_eps logcdf(d, 0.0874216) -0.916291 1e-5
@test_approx_eq_eps ccdf(d, 0.0874216) 0.6 1e-5
@test_approx_eq_eps logccdf(d, 0.0874216) -0.510826 1e-5
@test_approx_eq_eps pdf(d, 0.366513) 0.346574 1e-5
@test_approx_eq_eps cdf(d, 0.366513) 0.5 1e-5
@test_approx_eq_eps logpdf(d, 0.366513) -1.05966 1e-5
@test_approx_eq_eps logcdf(d, 0.366513) -0.693147 1e-5
@test_approx_eq_eps ccdf(d, 0.366513) 0.5 1e-5
@test_approx_eq_eps logccdf(d, 0.366513) -0.693147 1e-5
@test_approx_eq_eps pdf(d, 0.671727) 0.306495 1e-5
@test_approx_eq_eps cdf(d, 0.671727) 0.6 1e-5
@test_approx_eq_eps logpdf(d, 0.671727) -1.18255 1e-5
@test_approx_eq_eps logcdf(d, 0.671727) -0.510826 1e-5
@test_approx_eq_eps ccdf(d, 0.671727) 0.4 1e-5
@test_approx_eq_eps logccdf(d, 0.671727) -0.916291 1e-5
@test_approx_eq_eps pdf(d, 1.03093) 0.249672 1e-5
@test_approx_eq_eps cdf(d, 1.03093) 0.7 1e-5
@test_approx_eq_eps logpdf(d, 1.03093) -1.38761 1e-5
@test_approx_eq_eps logcdf(d, 1.03093) -0.356675 1e-5
@test_approx_eq_eps ccdf(d, 1.03093) 0.3 1e-5
@test_approx_eq_eps logccdf(d, 1.03093) -1.20397 1e-5
@test_approx_eq_eps pdf(d, 1.49994) 0.178515 1e-5
@test_approx_eq_eps cdf(d, 1.49994) 0.8 1e-5
@test_approx_eq_eps logpdf(d, 1.49994) -1.72308 1e-5
@test_approx_eq_eps logcdf(d, 1.49994) -0.223144 1e-5
@test_approx_eq_eps ccdf(d, 1.49994) 0.2 1e-5
@test_approx_eq_eps logccdf(d, 1.49994) -1.60944 1e-5
@test_approx_eq_eps pdf(d, 2.25037) 0.0948245 1e-5
@test_approx_eq_eps cdf(d, 2.25037) 0.9 1e-5
@test_approx_eq_eps logpdf(d, 2.25037) -2.35573 1e-5
@test_approx_eq_eps logcdf(d, 2.25037) -0.105361 1e-5
@test_approx_eq_eps ccdf(d, 2.25037) 0.1 1e-5
@test_approx_eq_eps logccdf(d, 2.25037) -2.30259 1e-5
@test_approx_eq_eps quantile(d, 0.1) -0.834032 1e-5
@test_approx_eq_eps quantile(d, 0.25) -0.326634 1e-5
@test_approx_eq_eps quantile(d, 0.5) 0.366513 1e-5
@test_approx_eq_eps quantile(d, 0.75) 1.2459 1e-5
@test_approx_eq_eps quantile(d, 0.9) 2.25037 1e-5

# test[0., 1., 1.1]
d = GeneralizedExtremeValue(0., 1., 1.1)
@test_approx_eq location(d) 0.
@test_approx_eq scale(d) 1.
@test_approx_eq shape(d) 1.1
@test_approx_eq_eps maximum(d) Inf 1e-5
@test_approx_eq_eps minimum(d) -0.909091 1e-5
@test_approx_eq_eps mean(d) Inf 1e-5
@test_approx_eq_eps median(d) 0.451411 1e-5
@test_approx_eq_eps var(d) Inf 1e-5
@test_approx_eq_eps skewness(d) Inf 1e-5
@test_approx_eq_eps kurtosis(d) Inf 1e-5
@test_approx_eq_eps pdf(d, -2.43627) 0. 1e-5
@test_approx_eq_eps cdf(d, -2.43627) 0. 1e-5
@test_approx_eq_eps logpdf(d, -2.43627) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -2.43627) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -2.43627) 1. 1e-5
@test_approx_eq_eps logccdf(d, -2.43627) 0. 1e-5
@test_approx_eq_eps pdf(d, -1.75017) 0. 1e-5
@test_approx_eq_eps cdf(d, -1.75017) 0. 1e-5
@test_approx_eq_eps logpdf(d, -1.75017) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -1.75017) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -1.75017) 1. 1e-5
@test_approx_eq_eps logccdf(d, -1.75017) 0. 1e-5
@test_approx_eq_eps pdf(d, -1.06408) 0. 1e-5
@test_approx_eq_eps cdf(d, -1.06408) 0. 1e-5
@test_approx_eq_eps logpdf(d, -1.06408) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -1.06408) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -1.06408) 1. 1e-5
@test_approx_eq_eps logccdf(d, -1.06408) 0. 1e-5
@test_approx_eq_eps pdf(d, 0.091763) 0.332912 1e-5
@test_approx_eq_eps cdf(d, 0.091763) 0.4 1e-5
@test_approx_eq_eps logpdf(d, 0.091763) -1.09988 1e-5
@test_approx_eq_eps logcdf(d, 0.091763) -0.916291 1e-5
@test_approx_eq_eps ccdf(d, 0.091763) 0.6 1e-5
@test_approx_eq_eps logccdf(d, 0.091763) -0.510826 1e-5
@test_approx_eq_eps pdf(d, 0.451411) 0.231581 1e-5
@test_approx_eq_eps cdf(d, 0.451411) 0.5 1e-5
@test_approx_eq_eps logpdf(d, 0.451411) -1.46282 1e-5
@test_approx_eq_eps logcdf(d, 0.451411) -0.693147 1e-5
@test_approx_eq_eps ccdf(d, 0.451411) 0.5 1e-5
@test_approx_eq_eps logccdf(d, 0.451411) -0.693147 1e-5
@test_approx_eq_eps pdf(d, 0.99421) 0.146394 1e-5
@test_approx_eq_eps cdf(d, 0.99421) 0.6 1e-5
@test_approx_eq_eps logpdf(d, 0.99421) -1.92145 1e-5
@test_approx_eq_eps logcdf(d, 0.99421) -0.510826 1e-5
@test_approx_eq_eps ccdf(d, 0.99421) 0.4 1e-5
@test_approx_eq_eps logccdf(d, 0.99421) -0.916291 1e-5
@test_approx_eq_eps pdf(d, 1.91649) 0.0803287 1e-5
@test_approx_eq_eps cdf(d, 1.91649) 0.7 1e-5
@test_approx_eq_eps logpdf(d, 1.91649) -2.52163 1e-5
@test_approx_eq_eps logcdf(d, 1.91649) -0.356675 1e-5
@test_approx_eq_eps ccdf(d, 1.91649) 0.3 1e-5
@test_approx_eq_eps logccdf(d, 1.91649) -1.20397 1e-5
@test_approx_eq_eps pdf(d, 3.82421) 0.034286 1e-5
@test_approx_eq_eps cdf(d, 3.82421) 0.8 1e-5
@test_approx_eq_eps logpdf(d, 3.82421) -3.37302 1e-5
@test_approx_eq_eps logcdf(d, 3.82421) -0.223144 1e-5
@test_approx_eq_eps ccdf(d, 3.82421) 0.2 1e-5
@test_approx_eq_eps logccdf(d, 3.82421) -1.60944 1e-5
@test_approx_eq_eps pdf(d, 9.89683) 0.00797749 1e-5
@test_approx_eq_eps cdf(d, 9.89683) 0.9 1e-5
@test_approx_eq_eps logpdf(d, 9.89683) -4.83113 1e-5
@test_approx_eq_eps logcdf(d, 9.89683) -0.105361 1e-5
@test_approx_eq_eps ccdf(d, 9.89683) 0.1 1e-5
@test_approx_eq_eps logccdf(d, 9.89683) -2.30259 1e-5
@test_approx_eq_eps quantile(d, 0.1) -0.545871 1e-5
@test_approx_eq_eps quantile(d, 0.25) -0.274394 1e-5
@test_approx_eq_eps quantile(d, 0.5) 0.451411 1e-5
@test_approx_eq_eps quantile(d, 0.75) 2.67025 1e-5
@test_approx_eq_eps quantile(d, 0.9) 9.89683 1e-5

# test[0., 1., .6]
d = GeneralizedExtremeValue(0., 1., 0.6)
@test_approx_eq location(d) 0.
@test_approx_eq scale(d) 1.
@test_approx_eq shape(d) 0.6
@test_approx_eq_eps maximum(d) Inf 1e-5
@test_approx_eq_eps minimum(d) -1.66667 1e-5
@test_approx_eq_eps mean(d) 2.03027 1e-5
@test_approx_eq_eps median(d) 0.409936 1e-5
@test_approx_eq_eps var(d) Inf 1e-5
@test_approx_eq_eps skewness(d) Inf 1e-5
@test_approx_eq_eps kurtosis(d) Inf 1e-5
@test_approx_eq_eps pdf(d, -1.94901) 0. 1e-5
@test_approx_eq_eps cdf(d, -1.94901) 0. 1e-5
@test_approx_eq_eps logpdf(d, -1.94901) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -1.94901) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -1.94901) 1. 1e-5
@test_approx_eq_eps logccdf(d, -1.94901) 0. 1e-5
@test_approx_eq_eps pdf(d, -0.413975) 0.428261 1e-5
@test_approx_eq_eps cdf(d, -0.413975) 0.2 1e-5
@test_approx_eq_eps logpdf(d, -0.413975) -0.848022 1e-5
@test_approx_eq_eps logcdf(d, -0.413975) -1.60944 1e-5
@test_approx_eq_eps ccdf(d, -0.413975) 0.8 1e-5
@test_approx_eq_eps logccdf(d, -0.413975) -0.223144 1e-5
@test_approx_eq_eps pdf(d, -0.175663) 0.403746 1e-5
@test_approx_eq_eps cdf(d, -0.175663) 0.3 1e-5
@test_approx_eq_eps logpdf(d, -0.175663) -0.90697 1e-5
@test_approx_eq_eps logcdf(d, -0.175663) -1.20397 1e-5
@test_approx_eq_eps ccdf(d, -0.175663) 0.7 1e-5
@test_approx_eq_eps logccdf(d, -0.175663) -0.356675 1e-5
@test_approx_eq_eps pdf(d, 0.0897549) 0.347787 1e-5
@test_approx_eq_eps cdf(d, 0.0897549) 0.4 1e-5
@test_approx_eq_eps logpdf(d, 0.0897549) -1.05617 1e-5
@test_approx_eq_eps logcdf(d, 0.0897549) -0.916291 1e-5
@test_approx_eq_eps ccdf(d, 0.0897549) 0.6 1e-5
@test_approx_eq_eps logccdf(d, 0.0897549) -0.510826 1e-5
@test_approx_eq_eps pdf(d, 0.409936) 0.278157 1e-5
@test_approx_eq_eps cdf(d, 0.409936) 0.5 1e-5
@test_approx_eq_eps logpdf(d, 0.409936) -1.27957 1e-5
@test_approx_eq_eps logcdf(d, 0.409936) -0.693147 1e-5
@test_approx_eq_eps ccdf(d, 0.409936) 0.5 1e-5
@test_approx_eq_eps logccdf(d, 0.409936) -0.693147 1e-5
@test_approx_eq_eps pdf(d, 0.827268) 0.204827 1e-5
@test_approx_eq_eps cdf(d, 0.827268) 0.6 1e-5
@test_approx_eq_eps logpdf(d, 0.827268) -1.58559 1e-5
@test_approx_eq_eps logcdf(d, 0.827268) -0.510826 1e-5
@test_approx_eq_eps ccdf(d, 0.827268) 0.4 1e-5
@test_approx_eq_eps logccdf(d, 0.827268) -0.916291 1e-5
@test_approx_eq_eps pdf(d, 1.42708) 0.134504 1e-5
@test_approx_eq_eps cdf(d, 1.42708) 0.7 1e-5
@test_approx_eq_eps logpdf(d, 1.42708) -2.00616 1e-5
@test_approx_eq_eps logcdf(d, 1.42708) -0.356675 1e-5
@test_approx_eq_eps ccdf(d, 1.42708) 0.3 1e-5
@test_approx_eq_eps logccdf(d, 1.42708) -1.20397 1e-5
@test_approx_eq_eps pdf(d, 2.43252) 0.0725813 1e-5
@test_approx_eq_eps cdf(d, 2.43252) 0.8 1e-5
@test_approx_eq_eps logpdf(d, 2.43252) -2.62305 1e-5
@test_approx_eq_eps logcdf(d, 2.43252) -0.223144 1e-5
@test_approx_eq_eps ccdf(d, 2.43252) 0.2 1e-5
@test_approx_eq_eps logccdf(d, 2.43252) -1.60944 1e-5
@test_approx_eq_eps pdf(d, 4.76379) 0.0245769 1e-5
@test_approx_eq_eps cdf(d, 4.76379) 0.9 1e-5
@test_approx_eq_eps logpdf(d, 4.76379) -3.70595 1e-5
@test_approx_eq_eps logcdf(d, 4.76379) -0.105361 1e-5
@test_approx_eq_eps ccdf(d, 4.76379) 0.1 1e-5
@test_approx_eq_eps logccdf(d, 4.76379) -2.30259 1e-5
@test_approx_eq_eps quantile(d, 0.1) -0.656206 1e-5
@test_approx_eq_eps quantile(d, 0.25) -0.29662 1e-5
@test_approx_eq_eps quantile(d, 0.5) 0.409936 1e-5
@test_approx_eq_eps quantile(d, 0.75) 1.853 1e-5
@test_approx_eq_eps quantile(d, 0.9) 4.76379 1e-5

# test[0., 1., .3]
d = GeneralizedExtremeValue(0., 1., 0.3)
@test_approx_eq location(d) 0.
@test_approx_eq scale(d) 1.
@test_approx_eq shape(d) 0.3
@test_approx_eq_eps maximum(d) Inf 1e-5
@test_approx_eq_eps minimum(d) -3.33333 1e-5
@test_approx_eq_eps mean(d) 0.993518 1e-5
@test_approx_eq_eps median(d) 0.387422 1e-5
@test_approx_eq_eps var(d) 5.92458 1e-5
@test_approx_eq_eps skewness(d) 13.4836 5e-5 # Changed precision!
@test_approx_eq_eps kurtosis(d) Inf 1e-5
@test_approx_eq_eps pdf(d, -0.737875) 0.29572 1e-5
@test_approx_eq_eps cdf(d, -0.737875) 0.1 1e-5
@test_approx_eq_eps logpdf(d, -0.737875) -1.21834 1e-5
@test_approx_eq_eps logcdf(d, -0.737875) -2.30259 1e-5
@test_approx_eq_eps ccdf(d, -0.737875) 0.9 1e-5
@test_approx_eq_eps logccdf(d, -0.737875) -0.105361 1e-5
@test_approx_eq_eps pdf(d, -0.443476) 0.371284 1e-5
@test_approx_eq_eps cdf(d, -0.443476) 0.2 1e-5
@test_approx_eq_eps logpdf(d, -0.443476) -0.990787 1e-5
@test_approx_eq_eps logcdf(d, -0.443476) -1.60944 1e-5
@test_approx_eq_eps ccdf(d, -0.443476) 0.8 1e-5
@test_approx_eq_eps logccdf(d, -0.443476) -0.223144 1e-5
@test_approx_eq_eps pdf(d, -0.180553) 0.381877 1e-5
@test_approx_eq_eps cdf(d, -0.180553) 0.3 1e-5
@test_approx_eq_eps logpdf(d, -0.180553) -0.962658 1e-5
@test_approx_eq_eps logcdf(d, -0.180553) -1.20397 1e-5
@test_approx_eq_eps ccdf(d, -0.180553) 0.7 1e-5
@test_approx_eq_eps logccdf(d, -0.180553) -0.356675 1e-5
@test_approx_eq_eps pdf(d, 0.088578) 0.357029 1e-5
@test_approx_eq_eps cdf(d, 0.088578) 0.4 1e-5
@test_approx_eq_eps logpdf(d, 0.088578) -1.02994 1e-5
@test_approx_eq_eps logcdf(d, 0.088578) -0.916291 1e-5
@test_approx_eq_eps ccdf(d, 0.088578) 0.6 1e-5
@test_approx_eq_eps logccdf(d, 0.088578) -0.510826 1e-5
@test_approx_eq_eps pdf(d, 0.387422) 0.310487 1e-5
@test_approx_eq_eps cdf(d, 0.387422) 0.5 1e-5
@test_approx_eq_eps logpdf(d, 0.387422) -1.16961 1e-5
@test_approx_eq_eps logcdf(d, 0.387422) -0.693147 1e-5
@test_approx_eq_eps ccdf(d, 0.387422) 0.5 1e-5
@test_approx_eq_eps logccdf(d, 0.387422) -0.693147 1e-5
@test_approx_eq_eps pdf(d, 0.744195) 0.250557 1e-5
@test_approx_eq_eps cdf(d, 0.744195) 0.6 1e-5
@test_approx_eq_eps logpdf(d, 0.744195) -1.38407 1e-5
@test_approx_eq_eps logcdf(d, 0.744195) -0.510826 1e-5
@test_approx_eq_eps ccdf(d, 0.744195) 0.4 1e-5
@test_approx_eq_eps logccdf(d, 0.744195) -0.916291 1e-5
@test_approx_eq_eps pdf(d, 1.20814) 0.183254 1e-5
@test_approx_eq_eps cdf(d, 1.20814) 0.7 1e-5
@test_approx_eq_eps logpdf(d, 1.20814) -1.69688 1e-5
@test_approx_eq_eps logcdf(d, 1.20814) -0.356675 1e-5
@test_approx_eq_eps ccdf(d, 1.20814) 0.3 1e-5
@test_approx_eq_eps logccdf(d, 1.20814) -1.20397 1e-5
@test_approx_eq_eps pdf(d, 1.89428) 0.113828 1e-5
@test_approx_eq_eps cdf(d, 1.89428) 0.8 1e-5
@test_approx_eq_eps logpdf(d, 1.89428) -2.17307 1e-5
@test_approx_eq_eps logcdf(d, 1.89428) -0.223144 1e-5
@test_approx_eq_eps ccdf(d, 1.89428) 0.2 1e-5
@test_approx_eq_eps logccdf(d, 1.89428) -1.60944 1e-5
@test_approx_eq_eps pdf(d, 3.21416) 0.0482752 1e-5
@test_approx_eq_eps cdf(d, 3.21416) 0.9 1e-5
@test_approx_eq_eps logpdf(d, 3.21416) -3.03084 1e-5
@test_approx_eq_eps logcdf(d, 3.21416) -0.105361 1e-5
@test_approx_eq_eps ccdf(d, 3.21416) 0.1 1e-5
@test_approx_eq_eps logccdf(d, 3.21416) -2.30259 1e-5
@test_approx_eq_eps quantile(d, 0.1) -0.737875 1e-5
@test_approx_eq_eps quantile(d, 0.25) -0.311141 1e-5
@test_approx_eq_eps quantile(d, 0.5) 0.387422 1e-5
@test_approx_eq_eps quantile(d, 0.75) 1.51068 1e-5
@test_approx_eq_eps quantile(d, 0.9) 3.21416 1e-5

# test[1., 1., -1.]
d = GeneralizedExtremeValue(1., 1., -1.)
@test_approx_eq location(d) 1.
@test_approx_eq scale(d) 1.
@test_approx_eq shape(d) -1.
@test_approx_eq_eps maximum(d) 2. 1e-5
@test_approx_eq_eps minimum(d) -Inf 1e-5
@test_approx_eq_eps mean(d) 1. 1e-5
@test_approx_eq_eps median(d) 1.30685 1e-5
@test_approx_eq_eps var(d) 1. 1e-5
@test_approx_eq_eps skewness(d) -2. 1e-5
@test_approx_eq_eps kurtosis(d) 6. 1e-5
@test_approx_eq_eps pdf(d, -0.302585) 0.1 1e-5
@test_approx_eq_eps cdf(d, -0.302585) 0.1 1e-5
@test_approx_eq_eps logpdf(d, -0.302585) -2.30259 1e-5
@test_approx_eq_eps logcdf(d, -0.302585) -2.30259 1e-5
@test_approx_eq_eps ccdf(d, -0.302585) 0.9 1e-5
@test_approx_eq_eps logccdf(d, -0.302585) -0.105361 1e-5
@test_approx_eq_eps pdf(d, 0.390562) 0.2 1e-5
@test_approx_eq_eps cdf(d, 0.390562) 0.2 1e-5
@test_approx_eq_eps logpdf(d, 0.390562) -1.60944 1e-5
@test_approx_eq_eps logcdf(d, 0.390562) -1.60944 1e-5
@test_approx_eq_eps ccdf(d, 0.390562) 0.8 1e-5
@test_approx_eq_eps logccdf(d, 0.390562) -0.223144 1e-5
@test_approx_eq_eps pdf(d, 0.796027) 0.3 1e-5
@test_approx_eq_eps cdf(d, 0.796027) 0.3 1e-5
@test_approx_eq_eps logpdf(d, 0.796027) -1.20397 1e-5
@test_approx_eq_eps logcdf(d, 0.796027) -1.20397 1e-5
@test_approx_eq_eps ccdf(d, 0.796027) 0.7 1e-5
@test_approx_eq_eps logccdf(d, 0.796027) -0.356675 1e-5
@test_approx_eq_eps pdf(d, 1.08371) 0.4 1e-5
@test_approx_eq_eps cdf(d, 1.08371) 0.4 1e-5
@test_approx_eq_eps logpdf(d, 1.08371) -0.916291 1e-5
@test_approx_eq_eps logcdf(d, 1.08371) -0.916291 1e-5
@test_approx_eq_eps ccdf(d, 1.08371) 0.6 1e-5
@test_approx_eq_eps logccdf(d, 1.08371) -0.510826 1e-5
@test_approx_eq_eps pdf(d, 1.30685) 0.5 1e-5
@test_approx_eq_eps cdf(d, 1.30685) 0.5 1e-5
@test_approx_eq_eps logpdf(d, 1.30685) -0.693147 1e-5
@test_approx_eq_eps logcdf(d, 1.30685) -0.693147 1e-5
@test_approx_eq_eps ccdf(d, 1.30685) 0.5 1e-5
@test_approx_eq_eps logccdf(d, 1.30685) -0.693147 1e-5
@test_approx_eq_eps pdf(d, 1.48917) 0.6 1e-5
@test_approx_eq_eps cdf(d, 1.48917) 0.6 1e-5
@test_approx_eq_eps logpdf(d, 1.48917) -0.510826 1e-5
@test_approx_eq_eps logcdf(d, 1.48917) -0.510826 1e-5
@test_approx_eq_eps ccdf(d, 1.48917) 0.4 1e-5
@test_approx_eq_eps logccdf(d, 1.48917) -0.916291 1e-5
@test_approx_eq_eps pdf(d, 1.64333) 0.7 1e-5
@test_approx_eq_eps cdf(d, 1.64333) 0.7 1e-5
@test_approx_eq_eps logpdf(d, 1.64333) -0.356675 1e-5
@test_approx_eq_eps logcdf(d, 1.64333) -0.356675 1e-5
@test_approx_eq_eps ccdf(d, 1.64333) 0.3 1e-5
@test_approx_eq_eps logccdf(d, 1.64333) -1.20397 5e-5 # Changed precision!
@test_approx_eq_eps pdf(d, 2.17463) 0. 1e-5
@test_approx_eq_eps cdf(d, 2.17463) 1. 1e-5
@test_approx_eq_eps logpdf(d, 2.17463) -Inf 1e-5
@test_approx_eq_eps logcdf(d, 2.17463) 0. 1e-5
@test_approx_eq_eps ccdf(d, 2.17463) 0. 1e-5
@test_approx_eq_eps logccdf(d, 2.17463) -Inf 1e-5
@test_approx_eq_eps pdf(d, 2.44645) 0. 1e-5
@test_approx_eq_eps cdf(d, 2.44645) 1. 1e-5
@test_approx_eq_eps logpdf(d, 2.44645) -Inf 1e-5
@test_approx_eq_eps logcdf(d, 2.44645) 0. 1e-5
@test_approx_eq_eps ccdf(d, 2.44645) 0. 1e-5
@test_approx_eq_eps logccdf(d, 2.44645) -Inf 1e-5
@test_approx_eq_eps quantile(d, 0.1) -0.302585 1e-5
@test_approx_eq_eps quantile(d, 0.25) 0.613706 1e-5
@test_approx_eq_eps quantile(d, 0.5) 1.30685 1e-5
@test_approx_eq_eps quantile(d, 0.75) 1.71232 1e-5
@test_approx_eq_eps quantile(d, 0.9) 1.89464 1e-5

# test[-1, 0.5, 0.6]
d = GeneralizedExtremeValue(-1., 0.5, 0.6)
@test_approx_eq location(d) -1.
@test_approx_eq scale(d) 0.5
@test_approx_eq shape(d) 0.6
@test_approx_eq_eps maximum(d) Inf 1e-5
@test_approx_eq_eps minimum(d) -1.83333 1e-5
@test_approx_eq_eps mean(d) 0.015133 1e-5
@test_approx_eq_eps median(d) -0.795032 1e-5
@test_approx_eq_eps var(d) Inf 1e-5
@test_approx_eq_eps skewness(d) Inf 1e-5
@test_approx_eq_eps kurtosis(d) Inf 1e-5
@test_approx_eq_eps pdf(d, -10.5807) 0. 1e-5
@test_approx_eq_eps cdf(d, -10.5807) 0. 1e-5
@test_approx_eq_eps logpdf(d, -10.5807) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -10.5807) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -10.5807) 1. 1e-5
@test_approx_eq_eps logccdf(d, -10.5807) 0. 1e-5
@test_approx_eq_eps pdf(d, -9.09221) 0. 1e-5
@test_approx_eq_eps cdf(d, -9.09221) 0. 1e-5
@test_approx_eq_eps logpdf(d, -9.09221) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -9.09221) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -9.09221) 1. 1e-5
@test_approx_eq_eps logccdf(d, -9.09221) 0. 1e-5
@test_approx_eq_eps pdf(d, -7.60374) 0. 1e-5
@test_approx_eq_eps cdf(d, -7.60374) 0. 1e-5
@test_approx_eq_eps logpdf(d, -7.60374) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -7.60374) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -7.60374) 1. 1e-5
@test_approx_eq_eps logccdf(d, -7.60374) 0. 1e-5
@test_approx_eq_eps pdf(d, -6.11528) 0. 1e-5
@test_approx_eq_eps cdf(d, -6.11528) 0. 1e-5
@test_approx_eq_eps logpdf(d, -6.11528) -Inf 1e-5
@test_approx_eq_eps logcdf(d, -6.11528) -Inf 1e-5
@test_approx_eq_eps ccdf(d, -6.11528) 1. 1e-5
@test_approx_eq_eps logccdf(d, -6.11528) 0. 1e-5
@test_approx_eq_eps pdf(d, -0.795032) 0.556315 1e-5
@test_approx_eq_eps cdf(d, -0.795032) 0.5 1e-5
@test_approx_eq_eps logpdf(d, -0.795032) -0.586421 1e-5
@test_approx_eq_eps logcdf(d, -0.795032) -0.693147 1e-5
@test_approx_eq_eps ccdf(d, -0.795032) 0.5 1e-5
@test_approx_eq_eps logccdf(d, -0.795032) -0.693147 1e-5
@test_approx_eq_eps pdf(d, -0.586366) 0.409654 1e-5
@test_approx_eq_eps cdf(d, -0.586366) 0.6 1e-5
@test_approx_eq_eps logpdf(d, -0.586366) -0.892442 1e-5
@test_approx_eq_eps logcdf(d, -0.586366) -0.510826 1e-5
@test_approx_eq_eps ccdf(d, -0.586366) 0.4 1e-5
@test_approx_eq_eps logccdf(d, -0.586366) -0.916291 1e-5
@test_approx_eq_eps pdf(d, -0.286458) 0.269007 1e-5
@test_approx_eq_eps cdf(d, -0.286458) 0.7 1e-5
@test_approx_eq_eps logpdf(d, -0.286458) -1.31302 1e-5
@test_approx_eq_eps logcdf(d, -0.286458) -0.356675 1e-5
@test_approx_eq_eps ccdf(d, -0.286458) 0.3 1e-5
@test_approx_eq_eps logccdf(d, -0.286458) -1.20397 1e-5
@test_approx_eq_eps pdf(d, 0.216262) 0.145163 1e-5
@test_approx_eq_eps cdf(d, 0.216262) 0.8 1e-5
@test_approx_eq_eps logpdf(d, 0.216262) -1.9299 1e-5
@test_approx_eq_eps logcdf(d, 0.216262) -0.223144 1e-5
@test_approx_eq_eps ccdf(d, 0.216262) 0.2 1e-5
@test_approx_eq_eps logccdf(d, 0.216262) -1.60944 1e-5
@test_approx_eq_eps pdf(d, 1.3819) 0.0491538 1e-5
@test_approx_eq_eps cdf(d, 1.3819) 0.9 1e-5
@test_approx_eq_eps logpdf(d, 1.3819) -3.0128 1e-5
@test_approx_eq_eps logcdf(d, 1.3819) -0.105361 1e-5
@test_approx_eq_eps ccdf(d, 1.3819) 0.1 1e-5
@test_approx_eq_eps logccdf(d, 1.3819) -2.30259 1e-5
@test_approx_eq_eps quantile(d, 0.1) -1.3281 1e-5
@test_approx_eq_eps quantile(d, 0.25) -1.14831 1e-5
@test_approx_eq_eps quantile(d, 0.5) -0.795032 1e-5
@test_approx_eq_eps quantile(d, 0.75) -0.0735019 1e-5
@test_approx_eq_eps quantile(d, 0.9) 1.3819 1e-5
