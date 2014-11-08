Distributions.jl Release Notes
================================

**Note:** We reached a relatively stable API at version *0.5.0*, and have maintained release notes since then.

Changes from v0.5 to v0.6
---------------------------

* Some bug fixes
* Add `CategoricalDirectSampler`.
* Add univariate von Mises distribution ([#223])
* Add Fréchet distribution ([#238])
* Add noncentral hypergeometric distribution ([#255])
* More functions for hypergeometric distribution (``support``, ``mode``, ``kurtosis``, ``skewness``, etc) ([#256])
* More efficient algorithm to sample from truncated distributions. ([#243])
* Various updates to make it work with both Julia 0.3 and 0.4 
* Improved implementation of ``show``. ([#290])
* A consistent testing framework for univariate distributions ([#291])
* Reimplement truncated distributions, fixing various bugs ([#295])
* Refactored type system for multivariate normal distributions, supporting zero-mean normal seamlessly and introducing common constructors. ([#296])
* Add triangular distribution ([#237])
* Reimplement von Mises distribution, fixing a few bugs ([#300])
* Reimplement von Mises-Fisher distribution, making it consistent with the common interface ([#302])
* Reimplement mixture models, improving efficiency, numerical stability, and the friendliness of the user interface. ([#303])
* Reimplement Wishart and InverseWishart distributions. They now support the use of positive definite matrices of arbitrary subtype of `AbstractPDMat`. ([#304])
* Add ``probs`` method for ``Categorical``, ``Multinomial``, and ``MixtureModel``.

[#238]: https://github.com/JuliaStats/Distributions.jl/pull/238
[#223]: https://github.com/JuliaStats/Distributions.jl/pull/223
[#237]: https://github.com/JuliaStats/Distributions.jl/pull/237
[#243]: https://github.com/JuliaStats/Distributions.jl/pull/243
[#255]: https://github.com/JuliaStats/Distributions.jl/pull/255
[#256]: https://github.com/JuliaStats/Distributions.jl/pull/256
[#290]: https://github.com/JuliaStats/Distributions.jl/pull/290
[#291]: https://github.com/JuliaStats/Distributions.jl/pull/291
[#295]: https://github.com/JuliaStats/Distributions.jl/pull/295
[#296]: https://github.com/JuliaStats/Distributions.jl/pull/296
[#300]: https://github.com/JuliaStats/Distributions.jl/pull/300
[#302]: https://github.com/JuliaStats/Distributions.jl/pull/302
[#303]: https://github.com/JuliaStats/Distributions.jl/pull/303
[#304]: https://github.com/JuliaStats/Distributions.jl/pull/304

