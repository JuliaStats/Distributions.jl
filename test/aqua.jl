using Distributions
using Test

import Aqua

@testset "Aqua" begin
    Aqua.test_all(Distributions)
end
