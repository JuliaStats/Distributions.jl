using Test
using Distributions: Categorical, kldivergence
@test kldivergence(Categorical([0.0, 0.1, 0.9]), Categorical([0.1, 0.1, 0.8])) ≥ 0
@test kldivergence(Categorical([0.0, 0.1, 0.9]), Categorical([0.1, 0.1, 0.8])) ≈
    kldivergence([0.0, 0.1, 0.9], [0.1, 0.1, 0.8])
