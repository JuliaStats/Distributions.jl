using Distributions
using Test

a = qqbuild(collect(1:10), collect(1:10))
b = qqbuild(1:10, 1:10)
c = qqbuild(view(collect(1:20), 1:10), view(collect(1:20), 1:10))
@test a.qx ≈ b.qx ≈ c.qx ≈ collect(1.0:10)
@test a.qy ≈ b.qy ≈ c.qy ≈ collect(1.0:10)


pp = Distributions.ppoints(10)
@test length(pp) == 10
@test minimum(pp) >= 0
@test maximum(pp) <= 1

a = qqbuild(collect(1:10), Uniform(1,10))
b = qqbuild(1:10, Uniform(1,10))
c = qqbuild(view(collect(1:20), 1:10), Uniform(1,10))
@test length(a.qy) == length(a.qx) == 10
@test a.qx ≈ b.qx ≈ c.qx ≈ a.qy ≈ b.qy ≈ c.qy

a = qqbuild(Uniform(1,10), collect(1:10))
b = qqbuild(Uniform(1,10), 1:10)
c = qqbuild(Uniform(1,10), view(collect(1:20), 1:10))
@test length(a.qy) == length(a.qx) == 10
@test a.qx ≈ b.qx ≈ c.qx ≈ a.qy ≈ b.qy ≈ c.qy

for n in 0:3
    a = qqbuild(rand(n), Uniform(0,1))
    @test length(a.qy) == length(a.qx) == n
end
