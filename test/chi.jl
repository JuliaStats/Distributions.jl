# see #1305
@test isfinite(mean(Chi(1000)))

# see #1497
@test pdf(Chi(3), -.5) == 0.0