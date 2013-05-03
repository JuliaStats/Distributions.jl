
my_tests = [
    "test/distributions.jl", 
    "test/utils.jl",
    "test/wisharts.jl", 
    "test/fit.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
