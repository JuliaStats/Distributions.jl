@testset "Generalized hyperbolic" begin
    Hyp(z, p=0, μ=0, σ=1, λ=1) = GeneralizedHyperbolic(Val(:locscale), z, p, μ, σ, λ)
    # Empirical characteristic function
    cf_empirical(samples::AbstractVector{<:Real}, t::Real) = mean(x->exp(1im * t * x), samples)

    # Generated in Wolfram Language:
    #
    # Hyp[z_, p_ : 0, m_ : 0, s_ : 1, \[Lambda]_ : 1] :=
    #  HyperbolicDistribution[\[Lambda], z*Sqrt[1 + p^2]/s, z*p/s, s, m]
    # props[d_] := N[{Mean@d, Variance@d, Skewness@d, Kurtosis[d]-Kurtosis[NormalDistribution[]]}, 30]
    #
    # In[]:= props@Hyp[6, 1, -5, 2, 8]

    # Out[]= {1.06319517080168381227975950313, \
    # 5.63790040251113774016546423535, 0.625851630297306083508792293920, \
    # 0.63622204658230960216273474948}
    distributions = [
        # No skewness, location, scale
        (
            d=Hyp(3/10),
            tmean=0,
            tvar=23.7192375345642910012435489189,
            tskew=0,
            tkurt=2.68056465465234820166864415655, # EXCESS kurtosis!!
            tmode=0
        ), (
            d=Hyp(3),
            tmean=0,
            tvar=0.510590348325343769505341346913,
            tskew=0,
            tkurt=0.88995291430008067418638784349,
            tmode=0
        ), (
            d=Hyp(10),
            tmean=0,
            tvar=0.115341725074794522009169334735,
            tskew=0,
            tkurt=0.29539619644673402834977400441,
            tmode=0
        ),
        # Add skewness
        (
            d=Hyp(1/10, -5),
            tmean=-101.231534024878145278158003846,
            tvar=10225.9463919953237264034118170,
            tskew=-1.99460167233720176630548735540,
            tkurt=5.97343971844269169149962193657,
            tmode=-5
        ), (
            d=Hyp(3, -1),
            tmean=-1.53177104497603130851602404074,
            tvar=1.20662920739975591772951714814,
            tskew=-1.14677758937397095739187195816,
            tkurt=2.35384523161163514880079584352,
            tmode=-1
        ), (
            d=Hyp(8, 1),
            tmean=1.19272670576874667902532489770,
            tvar=0.322857196451500261892719850967,
            tskew=0.744755964647115580796052901480,
            tkurt=1.05145730182810368503382261950,
            tmode=1
        ), (
            d=Hyp(20, 5),
            tmean=5.37946731533635375645988863585,
            tvar=1.49459339171759999060447162252,
            tskew=0.641583383199309715147192149775,
            tkurt=0.68419269261165181164749648079,
            tmode=5
        ),
        # Add location & scale
        (
            d=Hyp(1, -2, -1, 5),
            tmean=-27.9948393559377234389267739977,
            tvar=518.559320774469042942015734302,
            tskew=-1.75030004538948425548612197692,
            tkurt=4.77694919849117624423554369362,
            tmode=-11
        ), (
            d=Hyp(1, -2, 1, 5),
            tmean=-25.9948393559377234389267739977,
            tvar=518.559320774469042942015734302,
            tskew=-1.75030004538948425548612197692,
            tkurt=4.77694919849117624423554369362,
            tmode=-9
        ), (
            d=Hyp(6, 1, -5, 2),
            tmean=-2.48204071280501662885409776329,
            tvar=1.85647984002017020249703380834,
            tskew=0.852297903911637048654316184661,
            tkurt=1.35763888237097731740356582167,
            tmode=-3
        ),
        # Different λ
        (
            d=Hyp(3/10, 0,0,1, -5),
            tmean=0,
            tvar=0.124766926919973221644770566113,
            tskew=0,
            tkurt=0.99265597462752447495230230680,
            tmode=0
        ), (
            d=Hyp(3, 0,0,1, -1),
            tmean=0,
            tvar=0.288368126103121547283119124690,
            tskew=0,
            tkurt=1.00852166077397394811681109436,
            tmode=0
        ), (
            d=Hyp(10, 0,0,1, 4),
            tmean=0,
            tvar=0.151980099806535291929364762119,
            tskew=0,
            tkurt=0.27275916387679590474273501914,
            tmode=0
        ), (
            d=Hyp(3, -1, 0,1, -1),
            tmean=-0.865104378309364641849357374071,
            tvar=0.539962540733089251062850481473,
            tskew=-1.21718592500810682678970904231,
            tkurt=3.11624839625576195768292775000,
            tmode=-0.560464077632561342794140059659
        ), (
            d=Hyp(8, 1, 0,1, 5),
            tmean=1.84954386398016460385917258321,
            tvar=0.584696274181089855715315261983,
            tskew=0.668226904594935516683465663351,
            tkurt=0.76850406887218305023668985068,
            tmode=1.60212984628092067584739859051
        ), (
            d=Hyp(1, -2, -1, 5, -1/2),
            tmean=-11,
            tvar=125,
            tskew=-2.68328157299974763569100840248,
            tkurt=12.6,
            tmode=-4.39143813989001345842141045503
        ), (
            d=Hyp(1, -2, -1, 5, 1/2),
            tmean=-21,
            tvar=350,
            tskew=-2.02354940305121319832590492666,
            tkurt=6.53571428571428571428571428571,
            tmode=-7.49187863276682628498920382058
        ), (
            d=Hyp(6, 1, -5, 2, 8),
            tmean=1.06319517080168381227975950313,
            tvar=5.63790040251113774016546423535,
            tskew=0.625851630297306083508792293920,
            tkurt=0.63622204658230960216273474948,
            tmode=0.327863171219295387089685071504
        ),
    ]
    NSAMPLES = 10^6
    for (d, mean_true, var_true, skew_true, kurt_true, mode_true) in distributions
        println("\ttesting $d")

        @test collect(params(d)) ≈ [d.α, d.β, d.δ, d.μ, d.λ]

        # Specified `atol` because sometimes ground truth is zero, but we may get some floating-point inaccuracy
        @test mean(d)     ≈ mean_true atol=1e-6
        @test var(d)      ≈ var_true  atol=1e-6
        @test skewness(d) ≈ skew_true atol=1e-6
        @test kurtosis(d) ≈ kurt_true atol=1e-6
        @test mode(d)     ≈ mode_true atol=1e-6

        samples = rand(d, NSAMPLES)
        @test abs(mean(d) - mean(samples)) < 0.02
        @test abs(std(d) - std(samples)) < 0.05
        
        # Empirical CF should be close to theoretical CF
        @test maximum(t->abs(cf(d, t) - cf_empirical(samples, t)), range(-50, 50, 100)) < 0.005

        test_samples(d, NSAMPLES)
    end
end
