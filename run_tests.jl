
tests = [
    "types",
	"utils", 
	"sample",
	"fit", 
	"discrete",
	"univariate", 
	"truncate", 
	"multinomial",
	"dirichlet", 
	"mvnormal",
	"kolmogorov",
	"edgeworth",
	"matrix",
	"vonmisesfisher",
	"compoundvariate",
	"conjugates",
    "conjugates_normal",
    "conjugates_mvnormal"]

println("Running tests:")

for t in tests
	test_fn = joinpath("test", "$t.jl")
    println(" * $test_fn")
    include(test_fn)
end
