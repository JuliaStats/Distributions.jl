# see #1305
@test isfinite(mean(Chi(1000)))

# see #1497
for x in (-0.5, -1, -1.5f0), ν in (3, 3f0, 3.0)
    @test @inferred(logpdf(Chi(ν), x)) == -Inf
end 