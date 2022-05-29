using Test
using Distributions
using HypothesisTests: ExactOneSampleKSTest, pvalue


@testset "Trapezoidal" begin
    @test_throws ArgumentError TrapezoidalDist(2,1,3,4)
    @test_throws ArgumentError TrapezoidalDist(1,3,2,4)
    @test_throws ArgumentError TrapezoidalDist(1,2,4,3)

    d = TrapezoidalDist(1, 2, 3, 4)
    d2 = TrapezoidalDist(1.0f0, 2, 3, 4)
    @test partype(d) == Float64
    @test partype(d2) == Float32
    @test params(d) == (1.0, 2.0, 3.0, 4.0)
    @test minimum(d) ≈ 1
    @test maximum(d) ≈ 4

    @test logpdf(d, 3.3) ≈ log(pdf(d, 3.3))
    @test logpdf(d, 2) == logpdf(d, 3)
    # out of support
    @test isinf(logpdf(d, 0.5))
    @test isinf(logpdf(d, 4.5))
    @test cdf(d, 0.5) ≈ 0.0
    @test cdf(d, 4.5) ≈ 1.0
    # on support limits
    @test isinf(logpdf(d, 1))
    @test isinf(logpdf(d, 4))
    @test cdf(d, 1) ≈ 0.0
    @test cdf(d, 4) ≈ 1.0
    @test cdf(d, 2.5) ≈ 1/2

    @test mean(d) ≈ 2.5
    @test var(d) ≈ 0.41666666666666696

    # cdf and quantile
    for x in range(1, stop=4, length=20)
        @test quantile(d, cdf(d, x)) ≈ x
    end

    for p in range(0, stop=1, length=20)
        @test cdf(d, quantile(d, p)) ≈ p
    end


    # - compare with R package 'trapezoid' -
    # values generated with:
    # R version 4.0.1 (2020-06-06)
    # Platform: x86_64-pc-linux-gnu (64-bit)
    # Running under: Manjaro Linux
    # trapezoid_2.0-0

    # x <- seq(0, 6, 0.1)
    # dtrapezoid(x, 1, 2, 3, 5)
    dref  = [0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.04000000000000003552714,
             0.08000000000000007105427, 0.12000000000000002331468,
             0.16000000000000005884182, 0.20000000000000001110223,
             0.24000000000000004662937, 0.28000000000000008215650,
             0.32000000000000006217249, 0.36000000000000009769963,
             0.40000000000000002220446, 0.40000000000000002220446,
             0.40000000000000002220446, 0.40000000000000002220446,
             0.40000000000000002220446, 0.40000000000000002220446,
             0.40000000000000002220446, 0.40000000000000002220446,
             0.40000000000000002220446, 0.40000000000000002220446,
             0.40000000000000002220446, 0.38000000000000000444089,
             0.35999999999999998667732, 0.33999999999999996891376,
             0.31999999999999995115019, 0.30000000000000004440892,
             0.27999999999999997113420, 0.25999999999999995337063,
             0.23999999999999996336264, 0.21999999999999994559907,
             0.20000000000000001110223, 0.17999999999999991007194,
             0.15999999999999997557509, 0.14000000000000004107825,
             0.11999999999999994004796, 0.10000000000000000555112,
             0.07999999999999990452082, 0.05999999999999997002398,
             0.03999999999999986205479, 0.01999999999999993102739,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000, 0.00000000000000000000000,
             0.00000000000000000000000]

    # dtrapezoid(x, 1, 2, 3, 5, log=TRUE)
    dlogref = [-Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf,                      -3.2188758248681996754215,
               -2.5257286443082547222616, -2.1202635362000910035363,
               -1.8325814637483097691018, -1.6094379124341002817999,
               -1.4271163556401456062872, -1.2729656758128871096858,
               -1.1394342831883645938973, -1.0216512475319809993834,
               -0.9162907318741549955732, -0.9162907318741549955732,
               -0.9162907318741549955732, -0.9162907318741549955732,
               -0.9162907318741549955732, -0.9162907318741549955732,
               -0.9162907318741549955732, -0.9162907318741549955732,
               -0.9162907318741549955732, -0.9162907318741549955732,
               -0.9162907318741549955732, -0.9675840262617055875793,
               -1.0216512475319814434727, -1.0788096613719300176371,
               -1.1394342831883650379865, -1.2039728043259358969408,
               -1.2729656758128875537750, -1.3470736479666094442820,
               -1.4271163556401458283318, -1.5141277326297757355178,
               -1.6094379124341002817999, -1.7147984280919272848109,
               -1.8325814637483102131910, -1.9661128563728325069349,
               -2.1202635362000914476255, -2.3025850929940454570044,
               -2.5257286443082564986184, -2.8134107167600368448745,
               -3.2188758248682041163136, -3.9120230054281495135626,
               -Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf,                      -Inf,
               -Inf]

    # ptrapezoid(x, 1, 2, 3, 5)
    pref = [0.000000000000000000000000, 0.000000000000000000000000,
            0.000000000000000000000000, 0.000000000000000000000000,
            0.000000000000000000000000, 0.000000000000000000000000,
            0.000000000000000000000000, 0.000000000000000000000000,
            0.000000000000000000000000, 0.000000000000000000000000,
            0.000000000000000000000000, 0.002000000000000003511080,
            0.008000000000000014044321, 0.018000000000000005578871,
            0.032000000000000021482816, 0.050000000000000002775558,
            0.072000000000000022315483, 0.098000000000000059285910,
            0.128000000000000030420111, 0.162000000000000060618177,
            0.200000000000000011102230, 0.240000000000000046629367,
            0.280000000000000082156504, 0.320000000000000117683641,
            0.360000000000000153210777, 0.400000000000000022204460,
            0.440000000000000057731597, 0.480000000000000093258734,
            0.520000000000000128785871, 0.560000000000000164313008,
            0.599999999999999977795540, 0.639000000000000012434498,
            0.676000000000000045297099, 0.711000000000000076383344,
            0.744000000000000105693232, 0.775000000000000022204460,
            0.804000000000000047961635, 0.831000000000000071942452,
            0.856000000000000094146912, 0.879000000000000003552714,
            0.900000000000000022204460, 0.919000000000000039079850,
            0.936000000000000054178884, 0.950999999999999956479257,
            0.964000000000000079047879, 0.974999999999999977795540,
            0.983999999999999985789145, 0.990999999999999992006394,
            0.995999999999999996447286, 0.998999999999999999111822,
            1.000000000000000000000000, 1.000000000000000000000000,
            1.000000000000000000000000, 1.000000000000000000000000,
            1.000000000000000000000000, 1.000000000000000000000000,
            1.000000000000000000000000, 1.000000000000000000000000,
            1.000000000000000000000000, 1.000000000000000000000000,
            1.000000000000000000000000]

    # ptrapezoid(x, 1, 2, 3, 5, log=TRUE)
    plogref = [-Inf,                        -Inf,
               -Inf,                        -Inf,
               -Inf,                        -Inf,
               -Inf,                        -Inf,
               -Inf,                        -Inf,
               -Inf,                        -6.214608098422189641496516,
               -4.828313737302299735176803, -4.017383521085972297726130,
               -3.442019376182409828857089, -2.995732273553990854253470,
               -2.631089159966081503227997, -2.322787800311564510025164,
               -2.055725015062519478448166, -1.820158943749752511465090,
               -1.609437912434100281799942, -1.427116355640145606287206,
               -1.272965675812887109685789, -1.139434283188364371852686,
               -1.021651247531980999383450, -0.916290731874154995573178,
               -0.820980552069830116224125, -0.733969175080200209038139,
               -0.653926467406663713965997, -0.579818495252941823459025,
               -0.510825623765990721736330, -0.447850824604602237855033,
               -0.391562202939172876448026, -0.341082849178896085895474,
               -0.295714244149045069054438, -0.254892249628790035220760,
               -0.218156009803170625183100, -0.185125484126688777397618,
               -0.155484902840394845213723, -0.128970381296960062700308,
               -0.105360515657826281366027, -0.084469156626449964919701,
               -0.066139802504544945027654, -0.050241216436746789775203,
               -0.036663984371591358535358, -0.025317807984289897316188,
               -0.016129381929883643970181, -0.009040744652149070720304,
               -0.004008021397538821806172, -0.001000500333583534363566,
               0.000000000000000000000000,  0.000000000000000000000000,
               0.000000000000000000000000,  0.000000000000000000000000,
               0.000000000000000000000000,  0.000000000000000000000000,
               0.000000000000000000000000,  0.000000000000000000000000,
               0.000000000000000000000000,  0.000000000000000000000000,
               0.000000000000000000000000]

    # p <- seq(0, 1, 0.01)
    # qtrapezoid(p, 1, 2, 3, 5)
    qref = [1.000000000000000000000, 1.223606797749978936096,
            1.316227766016837996688, 1.387298334620741702139,
            1.447213595499957872192, 1.500000000000000000000,
            1.547722557505166074421, 1.591607978309961701768,
            1.632455532033675993375, 1.670820393249937030333,
            1.707106781186547461715, 1.741619848709566209521,
            1.774596669241483404278, 1.806225774829854913150,
            1.836660026534075562665, 1.866025403784438596588,
            1.894427190999915744385, 1.921954445729288751821,
            1.948683298050513768018, 1.974679434480896222937,
            2.000000000000000000000, 2.024999999999999911182,
            2.049999999999999822364, 2.075000000000000177636,
            2.100000000000000088818, 2.125000000000000000000,
            2.149999999999999911182, 2.174999999999999822364,
            2.200000000000000177636, 2.225000000000000088818,
            2.250000000000000000000, 2.274999999999999911182,
            2.299999999999999822364, 2.325000000000000177636,
            2.350000000000000088818, 2.375000000000000000000,
            2.399999999999999911182, 2.424999999999999822364,
            2.450000000000000177636, 2.475000000000000088818,
            2.500000000000000000000, 2.524999999999999911182,
            2.549999999999999822364, 2.575000000000000177636,
            2.600000000000000088818, 2.625000000000000000000,
            2.649999999999999911182, 2.674999999999999822364,
            2.699999999999999733546, 2.724999999999999644729,
            2.750000000000000000000, 2.774999999999999911182,
            2.799999999999999822364, 2.825000000000000177636,
            2.850000000000000088818, 2.875000000000000000000,
            2.899999999999999911182, 2.924999999999999822364,
            2.949999999999999733546, 2.974999999999999644729,
            3.000000000000000000000, 3.025158234186849703917,
            3.050641131038207554127, 3.076461593832865659692,
            3.102633403898972463963, 3.129171306613029557298,
            3.156091108541422940448, 3.183409787541505053809,
            3.211145618000168511230, 3.239318313834099072324,
            3.267949192431123250913, 3.297061363407359557698,
            3.326679946931848874669, 3.356832327484501554693,
            3.387548450340290173699, 3.418861169915810016562,
            3.450806661517033191444, 3.483424911189690220681,
            3.516760302580867580957, 3.550862325381056283646,
            3.585786437626905076570, 3.621595124790978115215,
            3.658359213500126383423, 3.696159518959470702271,
            3.735088935932648013249, 3.775255128608410615243,
            3.816784043380076596463, 3.859824574900861726690,
            3.904554884989667851158, 3.951191151829848813293,
            4.000000000000000000000, 4.051316701949486009937,
            4.105572809000084255615, 4.163339973465924437335,
            4.225403330758517483901, 4.292893218813452982374,
            4.367544467966324006625, 4.452277442494833259445,
            4.552786404500041683718, 4.683772233983162003312,
            5.000000000000000000000]

    x = 0:0.1:6
    d = TrapezoidalDist(1,2,3,5)
    for (i, xs) in enumerate(x)
        @test pdf(d, xs) ≈ dref[i]
        @test logpdf(d, xs) ≈ dlogref[i]
        @test cdf(d, xs) ≈ pref[i]
        @test logcdf(d, xs) ≈ plogref[i]
    end

    p = 0:0.01:1
    for (i, ps) in enumerate(p)
        @test quantile(d, ps) ≈ qref[i]
    end


    # - rand -
    n = 100_000
    α = 0.05
    kt = ExactOneSampleKSTest(rand(d, n), d)
    @test pvalue(kt) >= α

end
