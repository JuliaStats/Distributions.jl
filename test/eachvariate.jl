using ChainRulesTestUtils
using ChainRulesTestUtils: FiniteDifferences

# Without this, `to_vec` will also include the `axes` field of `EachVariate`.
function FiniteDifferences.to_vec(xs::Distributions.EachVariate{V}) where {V}
    vals, vals_from_vec = FiniteDifferences.to_vec(xs.parent)
    return vals, x -> Distributions.EachVariate{V}(vals_from_vec(x))
end

@testset "eachvariate.jl" begin
    @testset "ChainRules" begin
        xs = randn(2, 3, 4, 5)
        test_rrule(Distributions.EachVariate{1}, xs)
        test_rrule(Distributions.EachVariate{2}, xs)
        test_rrule(Distributions.EachVariate{3}, xs)
    end
end
