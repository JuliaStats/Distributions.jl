@testset "Pareto Sampling Tests" begin
    for d in [
            Pareto()
            Pareto(2.0)
            Pareto(2.0, 1.5)
            Pareto(3.0, 2.0)
        ]
        test_distr(d, 10^6)
    end
end
