using Distributions
using Test

a = qqbuild(collect(1:10), collect(1:10))
b = qqbuild(1:10, 1:10)
c = qqbuild(view(collect(1:20), 1:10), view(collect(1:20), 1:10))
@test a.qx ≈ b.qx ≈ c.qx ≈ collect(1.0:10)
@test a.qy ≈ b.qy ≈ c.qy ≈ collect(1.0:10)

a = qqbuild(collect(1:10), Uniform(1,10))
b = qqbuild(1:10, Uniform(1,10))
c = qqbuild(view(collect(1:20), 1:10), Uniform(1,10))
@test a.qx ≈ b.qx ≈ c.qx ≈ collect(2.0:9)
@test a.qy ≈ b.qy ≈ c.qy ≈ collect(2.0:9)

a = qqbuild(Uniform(1,10), collect(1:10))
b = qqbuild(Uniform(1,10), 1:10)
c = qqbuild(Uniform(1,10), view(collect(1:20), 1:10))
@test a.qx ≈ b.qx ≈ c.qx ≈ collect(2.0:9)
@test a.qy ≈ b.qy ≈ c.qy ≈ collect(2.0:9)
