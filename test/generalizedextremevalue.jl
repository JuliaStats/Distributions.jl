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
###    ent, max, min, mean, median, var,
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
###     <> "@test_approx_eq_eps maximum(d) " <> printv[max] <> " 1e-7\n"
###     <> "@test_approx_eq_eps minimum(d) " <> printv[min] <> " 1e-7\n"
###     <> If[ent != None, 
###      "@test_approx_eq_eps entropy(d) " <> printv[ent] <> " 1e-7\n", 
###      "", ""]
###     <> "@test_approx_eq_eps mean(d) " <> printv[mean] <> " 1e-7\n"
###     <> "@test_approx_eq_eps median(d) " <> printv[median] <> " 1e-7\n"
###     <> "@test_approx_eq_eps var(d) " <> printv[var] <> " 1e-7\n"
###     <> ((
###         "@test_approx_eq_eps pdf(d, " <> printv[points[[#]]] <> ") " <>
###           printv[pointsPDF[[#]]] <> " 1e-7\n"
###          <>
###          "@test_approx_eq_eps cdf(d, " <> printv[points[[#]]] <> 
###          ") " <> printv[pointsCDF[[#]]] <> " 1e-7\n"
###         ) & /@ Table[i, {i, 1, Length@points}])
###     <> ((
###         "@test_approx_eq_eps quantile(d, " <> printv[q[[#]]] <> ") " <>
###           printv[quantiles[[#]]] <> " 1e-7\n"
###         ) & /@ Table[i, {i, 1, Length@quantiles}]);
###   
###   Return[code];
###   ];

# test[1., 1., 1.]
d = GeneralizedExtremeValue(1.0, 1.0, 1.0)
@test_approx_eq location(d) 1.0
@test_approx_eq scale(d) 1.0
@test_approx_eq shape(d) 1.0
@test_approx_eq_eps maximum(d) Inf 1e-7
@test_approx_eq_eps minimum(d) 0.0 1e-7
@test_approx_eq_eps mean(d) Inf 1e-7
@test_approx_eq_eps median(d) 1.44269504 1e-7
@test_approx_eq_eps var(d) Inf 1e-7
@test_approx_eq_eps pdf(d, 0.43429448) 0.53018981 1e-7
@test_approx_eq_eps cdf(d, 0.43429448) 0.1 1e-7
@test_approx_eq_eps pdf(d, 0.62133493) 0.51805808 1e-7
@test_approx_eq_eps cdf(d, 0.62133493) 0.2 1e-7
@test_approx_eq_eps pdf(d, 0.83058355) 0.43486515 1e-7
@test_approx_eq_eps cdf(d, 0.83058355) 0.3 1e-7
@test_approx_eq_eps pdf(d, 1.09135667) 0.33583548 1e-7
@test_approx_eq_eps cdf(d, 1.09135667) 0.4 1e-7
@test_approx_eq_eps pdf(d, 1.44269504) 0.24022651 1e-7
@test_approx_eq_eps cdf(d, 1.44269504) 0.5 1e-7
@test_approx_eq_eps pdf(d, 1.95761519) 0.15656569 1e-7
@test_approx_eq_eps cdf(d, 1.95761519) 0.6 1e-7
@test_approx_eq_eps pdf(d, 2.80367325) 0.08905191 1e-7
@test_approx_eq_eps cdf(d, 2.80367325) 0.7 1e-7
@test_approx_eq_eps pdf(d, 4.48142012) 0.03983444 1e-7
@test_approx_eq_eps cdf(d, 4.48142012) 0.8 1e-7
@test_approx_eq_eps pdf(d, 9.49122158) 0.00999075 1e-7
@test_approx_eq_eps cdf(d, 9.49122158) 0.9 1e-7
@test_approx_eq_eps quantile(d, 0.1) 0.43429448 1e-7
@test_approx_eq_eps quantile(d, 0.25) 0.72134752 1e-7
@test_approx_eq_eps quantile(d, 0.5) 1.44269504 1e-7
@test_approx_eq_eps quantile(d, 0.75) 3.4760595 1e-7
@test_approx_eq_eps quantile(d, 0.9) 9.49122158 1e-7

# test[0., 1., 0.]
d = GeneralizedExtremeValue(0.0, 1.0, 0.0)
@test_approx_eq location(d) 0.0
@test_approx_eq scale(d) 1.0
@test_approx_eq shape(d) 0.0
@test_approx_eq_eps maximum(d) Inf 1e-7
@test_approx_eq_eps minimum(d) -Inf 1e-7
@test_approx_eq_eps mean(d) 0.57721566 1e-7
@test_approx_eq_eps median(d) 0.36651292 1e-7
@test_approx_eq_eps var(d) 1.64493407 1e-7
@test_approx_eq_eps pdf(d, -0.83403245) 0.23025851 1e-7
@test_approx_eq_eps cdf(d, -0.83403245) 0.1 1e-7
@test_approx_eq_eps pdf(d, -0.475885) 0.32188758 1e-7
@test_approx_eq_eps cdf(d, -0.475885) 0.2 1e-7
@test_approx_eq_eps pdf(d, -0.18562676) 0.36119184 1e-7
@test_approx_eq_eps cdf(d, -0.18562676) 0.3 1e-7
@test_approx_eq_eps pdf(d, 0.08742157) 0.36651629 1e-7
@test_approx_eq_eps cdf(d, 0.08742157) 0.4 1e-7
@test_approx_eq_eps pdf(d, 0.36651292) 0.34657359 1e-7
@test_approx_eq_eps cdf(d, 0.36651292) 0.5 1e-7
@test_approx_eq_eps pdf(d, 0.67172699) 0.30649537 1e-7
@test_approx_eq_eps cdf(d, 0.67172699) 0.6 1e-7
@test_approx_eq_eps pdf(d, 1.03093043) 0.24967246 1e-7
@test_approx_eq_eps cdf(d, 1.03093043) 0.7 1e-7
@test_approx_eq_eps pdf(d, 1.49993999) 0.17851484 1e-7
@test_approx_eq_eps cdf(d, 1.49993999) 0.8 1e-7
@test_approx_eq_eps pdf(d, 2.25036733) 0.09482446 1e-7
@test_approx_eq_eps cdf(d, 2.25036733) 0.9 1e-7
@test_approx_eq_eps quantile(d, 0.1) -0.83403245 1e-7
@test_approx_eq_eps quantile(d, 0.25) -0.32663426 1e-7
@test_approx_eq_eps quantile(d, 0.5) 0.36651292 1e-7
@test_approx_eq_eps quantile(d, 0.75) 1.24589932 1e-7
@test_approx_eq_eps quantile(d, 0.9) 2.25036733 1e-7

# test[0., 1., 1.1]
d = GeneralizedExtremeValue(0.0, 1.0, 1.1)
@test_approx_eq location(d) 0.0
@test_approx_eq scale(d) 1.0
@test_approx_eq shape(d) 1.1
@test_approx_eq_eps maximum(d) Inf 1e-7
@test_approx_eq_eps minimum(d) -0.90909091 1e-7
@test_approx_eq_eps mean(d) Inf 1e-7
@test_approx_eq_eps median(d) 0.45141148 1e-7
@test_approx_eq_eps var(d) Inf 1e-7
@test_approx_eq_eps pdf(d, -2.43626727) 0.0 1e-7
@test_approx_eq_eps cdf(d, -2.43626727) 0.0 1e-7
@test_approx_eq_eps pdf(d, -1.75017443) 0.0 1e-7
@test_approx_eq_eps cdf(d, -1.75017443) 0.0 1e-7
@test_approx_eq_eps pdf(d, -1.06408159) 0.0 1e-7
@test_approx_eq_eps cdf(d, -1.06408159) 0.0 1e-7
@test_approx_eq_eps pdf(d, 0.091763) 0.33291235 1e-7
@test_approx_eq_eps cdf(d, 0.091763) 0.4 1e-7
@test_approx_eq_eps pdf(d, 0.45141148) 0.23158129 1e-7
@test_approx_eq_eps cdf(d, 0.45141148) 0.5 1e-7
@test_approx_eq_eps pdf(d, 0.99420964) 0.1463942 1e-7
@test_approx_eq_eps cdf(d, 0.99420964) 0.6 1e-7
@test_approx_eq_eps pdf(d, 1.91648808) 0.08032866 1e-7
@test_approx_eq_eps cdf(d, 1.91648808) 0.7 1e-7
@test_approx_eq_eps pdf(d, 3.82421464) 0.03428602 1e-7
@test_approx_eq_eps cdf(d, 3.82421464) 0.8 1e-7
@test_approx_eq_eps pdf(d, 9.89682637) 0.00797749 1e-7
@test_approx_eq_eps cdf(d, 9.89682637) 0.9 1e-7
@test_approx_eq_eps quantile(d, 0.1) -0.54587066 1e-7
@test_approx_eq_eps quantile(d, 0.25) -0.2743941 1e-7
@test_approx_eq_eps quantile(d, 0.5) 0.45141148 1e-7
@test_approx_eq_eps quantile(d, 0.75) 2.67025142 1e-7
@test_approx_eq_eps quantile(d, 0.9) 9.89682637 1e-7

# test[0., 1., .6]
d = GeneralizedExtremeValue(0.0, 1.0, 0.6)
@test_approx_eq location(d) 0.0
@test_approx_eq scale(d) 1.0
@test_approx_eq shape(d) 0.6
@test_approx_eq_eps maximum(d) Inf 1e-7
@test_approx_eq_eps minimum(d) -1.66666667 1e-7
@test_approx_eq_eps mean(d) 2.03026591 1e-7
@test_approx_eq_eps median(d) 0.40993631 1e-7
@test_approx_eq_eps var(d) Inf 1e-7
@test_approx_eq_eps pdf(d, -1.94901251) 0.0 1e-7
@test_approx_eq_eps cdf(d, -1.94901251) 0.0 1e-7
@test_approx_eq_eps pdf(d, -0.41397493) 0.42826123 1e-7
@test_approx_eq_eps cdf(d, -0.41397493) 0.2 1e-7
@test_approx_eq_eps pdf(d, -0.17566289) 0.40374573 1e-7
@test_approx_eq_eps cdf(d, -0.17566289) 0.3 1e-7
@test_approx_eq_eps pdf(d, 0.08975495) 0.34778693 1e-7
@test_approx_eq_eps cdf(d, 0.08975495) 0.4 1e-7
@test_approx_eq_eps pdf(d, 0.40993631) 0.27815748 1e-7
@test_approx_eq_eps cdf(d, 0.40993631) 0.5 1e-7
@test_approx_eq_eps pdf(d, 0.82726842) 0.20482715 1e-7
@test_approx_eq_eps cdf(d, 0.82726842) 0.6 1e-7
@test_approx_eq_eps pdf(d, 1.42708314) 0.13450369 1e-7
@test_approx_eq_eps cdf(d, 1.42708314) 0.7 1e-7
@test_approx_eq_eps pdf(d, 2.43252425) 0.07258133 1e-7
@test_approx_eq_eps cdf(d, 2.43252425) 0.8 1e-7
@test_approx_eq_eps pdf(d, 4.76379298) 0.0245769 1e-7
@test_approx_eq_eps cdf(d, 4.76379298) 0.9 1e-7
@test_approx_eq_eps quantile(d, 0.1) -0.65620618 1e-7
@test_approx_eq_eps quantile(d, 0.25) -0.29661964 1e-7
@test_approx_eq_eps quantile(d, 0.5) 0.40993631 1e-7
@test_approx_eq_eps quantile(d, 0.75) 1.85299623 1e-7
@test_approx_eq_eps quantile(d, 0.9) 4.76379298 1e-7

# test[0., 1., .3]
d = GeneralizedExtremeValue(0.0, 1.0, 0.3)
@test_approx_eq location(d) 0.0
@test_approx_eq scale(d) 1.0
@test_approx_eq shape(d) 0.3
@test_approx_eq_eps maximum(d) Inf 1e-7
@test_approx_eq_eps minimum(d) -3.33333333 1e-7
@test_approx_eq_eps mean(d) 0.99351778 1e-7
@test_approx_eq_eps median(d) 0.38742195 1e-7
@test_approx_eq_eps var(d) 5.92457663 1e-7
@test_approx_eq_eps pdf(d, -0.73787513) 0.29571979 1e-7
@test_approx_eq_eps cdf(d, -0.73787513) 0.1 1e-7
@test_approx_eq_eps pdf(d, -0.44347551) 0.37128422 1e-7
@test_approx_eq_eps cdf(d, -0.44347551) 0.2 1e-7
@test_approx_eq_eps pdf(d, -0.18055279) 0.3818765 1e-7
@test_approx_eq_eps cdf(d, -0.18055279) 0.3 1e-7
@test_approx_eq_eps pdf(d, 0.08857804) 0.35702882 1e-7
@test_approx_eq_eps cdf(d, 0.08857804) 0.4 1e-7
@test_approx_eq_eps pdf(d, 0.38742195) 0.31048677 1e-7
@test_approx_eq_eps cdf(d, 0.38742195) 0.5 1e-7
@test_approx_eq_eps pdf(d, 0.74419458) 0.25055653 1e-7
@test_approx_eq_eps cdf(d, 0.74419458) 0.6 1e-7
@test_approx_eq_eps pdf(d, 1.20814205) 0.18325356 1e-7
@test_approx_eq_eps cdf(d, 1.20814205) 0.7 1e-7
@test_approx_eq_eps pdf(d, 1.89427983) 0.11382814 1e-7
@test_approx_eq_eps cdf(d, 1.89427983) 0.8 1e-7
@test_approx_eq_eps pdf(d, 3.21416474) 0.04827516 1e-7
@test_approx_eq_eps cdf(d, 3.21416474) 0.9 1e-7
@test_approx_eq_eps quantile(d, 0.1) -0.73787513 1e-7
@test_approx_eq_eps quantile(d, 0.25) -0.31114094 1e-7
@test_approx_eq_eps quantile(d, 0.5) 0.38742195 1e-7
@test_approx_eq_eps quantile(d, 0.75) 1.51067527 1e-7
@test_approx_eq_eps quantile(d, 0.9) 3.21416474 1e-7

# test[0., 1., .0]
d = GeneralizedExtremeValue(0.0, 1.0, 0.0)
@test_approx_eq location(d) 0.0
@test_approx_eq scale(d) 1.0
@test_approx_eq shape(d) 0.0
@test_approx_eq_eps maximum(d) Inf 1e-7
@test_approx_eq_eps minimum(d) -Inf 1e-7
@test_approx_eq_eps mean(d) 0.57721566 1e-7
@test_approx_eq_eps median(d) 0.36651292 1e-7
@test_approx_eq_eps var(d) 1.64493407 1e-7
@test_approx_eq_eps pdf(d, -0.83403245) 0.23025851 1e-7
@test_approx_eq_eps cdf(d, -0.83403245) 0.1 1e-7
@test_approx_eq_eps pdf(d, -0.475885) 0.32188758 1e-7
@test_approx_eq_eps cdf(d, -0.475885) 0.2 1e-7
@test_approx_eq_eps pdf(d, -0.18562676) 0.36119184 1e-7
@test_approx_eq_eps cdf(d, -0.18562676) 0.3 1e-7
@test_approx_eq_eps pdf(d, 0.08742157) 0.36651629 1e-7
@test_approx_eq_eps cdf(d, 0.08742157) 0.4 1e-7
@test_approx_eq_eps pdf(d, 0.36651292) 0.34657359 1e-7
@test_approx_eq_eps cdf(d, 0.36651292) 0.5 1e-7
@test_approx_eq_eps pdf(d, 0.67172699) 0.30649537 1e-7
@test_approx_eq_eps cdf(d, 0.67172699) 0.6 1e-7
@test_approx_eq_eps pdf(d, 1.03093043) 0.24967246 1e-7
@test_approx_eq_eps cdf(d, 1.03093043) 0.7 1e-7
@test_approx_eq_eps pdf(d, 1.49993999) 0.17851484 1e-7
@test_approx_eq_eps cdf(d, 1.49993999) 0.8 1e-7
@test_approx_eq_eps pdf(d, 2.25036733) 0.09482446 1e-7
@test_approx_eq_eps cdf(d, 2.25036733) 0.9 1e-7
@test_approx_eq_eps quantile(d, 0.1) -0.83403245 1e-7
@test_approx_eq_eps quantile(d, 0.25) -0.32663426 1e-7
@test_approx_eq_eps quantile(d, 0.5) 0.36651292 1e-7
@test_approx_eq_eps quantile(d, 0.75) 1.24589932 1e-7
@test_approx_eq_eps quantile(d, 0.9) 2.25036733 1e-7

# test[1., 1., -1.]
d = GeneralizedExtremeValue(1.0, 1.0, -1.0)
@test_approx_eq location(d) 1.0
@test_approx_eq scale(d) 1.0
@test_approx_eq shape(d) -1.0
@test_approx_eq_eps maximum(d) 2.0 1e-7
@test_approx_eq_eps minimum(d) -Inf 1e-7
@test_approx_eq_eps mean(d) 1.0 1e-7
@test_approx_eq_eps median(d) 1.30685282 1e-7
@test_approx_eq_eps var(d) 1.0 1e-7
@test_approx_eq_eps pdf(d, -0.30258509) 0.1 1e-7
@test_approx_eq_eps cdf(d, -0.30258509) 0.1 1e-7
@test_approx_eq_eps pdf(d, 0.39056209) 0.2 1e-7
@test_approx_eq_eps cdf(d, 0.39056209) 0.2 1e-7
@test_approx_eq_eps pdf(d, 0.7960272) 0.3 1e-7
@test_approx_eq_eps cdf(d, 0.7960272) 0.3 1e-7
@test_approx_eq_eps pdf(d, 1.08370927) 0.4 1e-7
@test_approx_eq_eps cdf(d, 1.08370927) 0.4 1e-7
@test_approx_eq_eps pdf(d, 1.30685282) 0.5 1e-7
@test_approx_eq_eps cdf(d, 1.30685282) 0.5 1e-7
@test_approx_eq_eps pdf(d, 1.48917438) 0.6 1e-7
@test_approx_eq_eps cdf(d, 1.48917438) 0.6 1e-7
@test_approx_eq_eps pdf(d, 1.64332506) 0.7 1e-7
@test_approx_eq_eps cdf(d, 1.64332506) 0.7 1e-7
@test_approx_eq_eps pdf(d, 2.17462546) 0.0 1e-7
@test_approx_eq_eps cdf(d, 2.17462546) 1.0 1e-7
@test_approx_eq_eps pdf(d, 2.44645365) 0.0 1e-7
@test_approx_eq_eps cdf(d, 2.44645365) 1.0 1e-7
@test_approx_eq_eps quantile(d, 0.1) -0.30258509 1e-7
@test_approx_eq_eps quantile(d, 0.25) 0.61370564 1e-7
@test_approx_eq_eps quantile(d, 0.5) 1.30685282 1e-7
@test_approx_eq_eps quantile(d, 0.75) 1.71231793 1e-7
@test_approx_eq_eps quantile(d, 0.9) 1.89463948 1e-7
