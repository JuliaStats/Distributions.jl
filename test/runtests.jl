
tests = [
    "types",
    "fit", 
    "discrete",
    "univariate", 
    "truncate", 
    "multinomial",
    "dirichlet",
    "multivariate",
    "mvnormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "vonmisesfisher",
    "compoundvariate",
    "conjugates",
    "conjugates_normal",
    "conjugates_mvnormal",
    "wishart",
    "mixture",
    "gradlogpdf",
    "triangular"]

println("Running tests:")

for t in tests
    test_fn = "$t.jl"
    println(" * $test_fn")
    include(test_fn)
end
