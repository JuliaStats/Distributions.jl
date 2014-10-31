tests = [
    "types",
    "samplers",
    "fit", 
    "discrete",
    "univariate", 
    "truncate", 
    "multinomial",
    "dirichlet",
    "hypergeometric",
    "multivariate",
    "mvnormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "noncentralhypergeometric",
    "vonmisesfisher",
    "compoundvariate",
    "conjugates",
    "conjugates_normal",
    "conjugates_mvnormal",
    "wishart",
    "mixture",
    "gradlogpdf"]

println("Running tests:")

for t in tests
    test_fn = "$t.jl"
    println(" * $test_fn")
    include(test_fn)
end
