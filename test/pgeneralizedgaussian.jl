@testset "generalized gaussian" begin

    d = GeneralizedGaussian() # mean zero, scale √2, shape 2.

    # PDF and CDF values from R using the same default
    # mean scale and shape parameters. Format is [x, pdf, cdf].
    test = [
        -2,0.053990966513188,0.0227501319481792;
        -1.9596,0.0584867753200998,0.0250212806912653;
        -1.9192,0.0632536242130714,0.0274795137708557;
        -1.8788,0.0682974223520507,0.0301359035293964;
        -1.8384,0.0736231464260287,0.0330017425665818;
        -1.798,0.0792347326126488,0.0360885038859181;
        -1.7576,0.0851349688324957,0.0394077966787751;
        -1.7172,0.091325388414829,0.0429713177801418;
        -1.6768,0.0978061663608371,0.0467907988768448;
        -1.6364,0.104576019449044,0.0508779495981413;
        -1.596,0.111632111473603,0.0552443966698443;
        -1.5556,0.118969964938241,0.0599016193659873;
        -1.5152,0.126583380545089,0.0648608815458657;
        -1.4748,0.134464365817196,0.0701331606184371;
        -1.4344,0.142603074175027,0.0757290738298424;
        -1.394,0.1509877557498,0.0816588023224809;
        -1.3536,0.159604721159404,0.0879320134647649;
        -1.3132,0.168438319395545,0.0945577819987581;
        -1.2728,0.177470930873585,0.101544510597386;
        -1.2324,0.186682976579551,0.108899850463094;
        -1.192,0.196052944112626,0.116630622634895;
        -1.1516,0.20555743126705,0.124742740699938;
        -1.1112,0.215171207626101,0.133241135628347;
        -1.0708,0.224867294454356,0.142129683465442;
        -1.0304,0.234617062974812,0.151411136623046;
        -0.99,0.244390350907,0.161087059510831;
        -0.9496,0.254155596923586,0.171157769239293;
        -0.9092,0.263879992459047,0.181622282107569;
        -0.8688,0.273529650077933,0.192478266561878;
        -0.8284,0.283069787385305,0.203722003273787;
        -0.788,0.292464925241515,0.215348352941949;
        -0.7476,0.301679098831105,0.227350732366586;
        -0.7072,0.31067607993471,0.239721099283345;
        -0.6668,0.319419608566778,0.252449946372615;
        -0.6264,0.327873631974065,0.265526304782692;
        -0.586,0.336002548843259,0.278937757421185;
        -0.5456,0.343771456443474,0.292670462179526;
        -0.5052,0.351146398333471,0.306709185161556;
        -0.4648,0.358094610196201,0.321037343889939;
        -0.4244,0.364584761326559,0.335637060364848;
        -0.384,0.370587189293274,0.350489223749189;
        -0.3436,0.376074125323367,0.36557356235494;
        -0.3032,0.38101990801804,0.380868724507227;
        -0.2628,0.385401183101656,0.396352367767935;
        -0.2224,0.389197087030118,0.412001255910218;
        -0.182,0.392389412439884,0.42779136295047;
        -0.1416,0.394962753602334,0.443697983466388;
        -0.1012,0.396904630257822,0.459695848359751;
        -0.0608,0.398205588436696,0.475759245161512;
        -0.0204,0.398859277127687,0.491862141925561;
        0.02,0.398862499923666,0.507978313716902;
        0.0604,0.398215241057149,0.524081470670474;
        0.1008,0.396920665528795,0.540145386578997;
        0.1412,0.394985093327439,0.556144026962162;
        0.1816,0.392417948035498,0.572051675575411;
        0.222,0.38923168040452,0.587843058334322;
        0.2624,0.385441667768147,0.603493463659977;
        0.3028,0.381066090429489,0.618978858291238;
        0.3432,0.376125786413223,0.634275997660943;
        0.3836,0.370644086205869,0.649362529994026;
        0.424,0.364646629317593,0.664217093355429;
        0.4644,0.358161164682611,0.678819404953587;
        0.5048,0.351217337070506,0.693150342089967;
        0.5452,0.343846461805699,0.707192014235536;
        0.5856,0.336081290185479,0.720927825809832;
        0.626,0.327955768047694,0.734342529336241;
        0.6664,0.319504789967085,0.747422268746731;
        0.7068,0.310763951554577,0.760154612709528;
        0.7472,0.301769302297434,0.772528577952478;
        0.7876,0.292557101311255,0.784534642652148;
        0.828,0.283163578279136,0.796164750052605;
        0.8684,0.273624701730973,0.807412302567402;
        0.9088,0.263975956669541,0.81827214670236;
        0.9492,0.254252133382257,0.828740549214512;
        0.9896,0.244487129091676,0.838815164993143;
        1.03,0.234713763897012,0.848494997211656;
        1.0704,0.224963612246627,0.857780350353393;
        1.1108,0.215266850961145,0.866672776760204;
        1.1512,0.205652124601902,0.875175017389245;
        1.1916,0.196146428753372,0.883290937490971;
        1.232,0.186775011564286,0.891025457939793;
        1.2724,0.177561293673426,0.898384482958291;
        1.3128,0.168526806435554,0.905374824976726;
        1.3532,0.159691148163201,0.912004127362079;
        1.3936,0.151071957913399,0.918280785735546;
        1.434,0.142684906177067,0.924213868574892;
        1.4744,0.13454370167416,0.929813037768908;
        1.5148,0.126660113321382,0.935088469756236;
        1.5552,0.119044006322032,0.940050777840661;
        1.5956,0.111703391230226,0.944710936230536;
        1.636,0.104644484764399,0.949080206301951;
        1.6764,0.0978717810877164,0.953170065534611;
        1.7168,0.091388132235354,0.956992139516678;
        1.7572,0.085194836349956,0.960558137361137;
        1.7976,0.0792917323859474,0.963879790822026;
        1.838,0.0736772999596982,0.966968797345075;
        1.8784,0.0683487630544158,0.969836767234444;
        1.9188,0.0633021963346127,0.97249517506594;
        1.9592,0.0585326328834392,0.97495531542798;
        1.9996,0.0540341722453821,0.977228263024933;
    ]

    # CDF test.
    for i=1:size(test, 1)
        @test isapprox(cdf(d, test[i, 1]), test[i, 3] ; atol = 1e-6)
    end

    # PDF test.
    for i=1:size(pdftest, 1)
        @test isapprox(pdf(d, test[i, 1]), test[i, 2] ; atol = 1e-6)
    end

    @test mean(d) == 0
    @test median(d) == 0
    @test mode(d) == 0
    @test var(d) == 1 # unity variance with shape 2 and scale √2
    @test std(d) == 1
    @test skewness(d) == 0
    @test kurtosis(d) ≈ 0
    @test entropy(d) ≈ 1.418938533204673

end
