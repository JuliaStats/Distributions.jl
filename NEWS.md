Distributions.jl Release Notes
================================

**Note:** We reached a relatively stable API at version *0.5.0*, and have maintained release notes since then.

Changes from v0.6 to v0.7
----------------------------
* Bug fixes
* Refactor file organization -- separate discrete and continuous distributions into different subdirectories
* Deprecate ``probs`` in favor of ``pdf``. Now ``pdf`` uses efficient algorithm to evaluate probability mass functions over a ``UnitRange``.
* Introduce macro ``@distr_supp``, which provides a uniform way to specify the support of a distribution, no matter whether the support depends on the distribution parameters or not. ([#312])
* New samplers for Gamma distribution. ([#313])
* New testing framework for univariate distributions. ([#314])
* Add requirement of the package ``Compat``. ([#321])
* Clean up the implementation of univariate distributions
* Add ``params`` methods and other parameter retrieval methods, such as ``scale``, ``shape``, ``meanlogx``, ``stdlogx``, ``dof``, etc ([#326])


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
* Add ``probs`` methods for discrete distributions ([#305]). 

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
[#305]: https://github.com/JuliaStats/Distributions.jl/pull/305
[#312]: https://github.com/JuliaStats/Distributions.jl/pull/312
[#313]: https://github.com/JuliaStats/Distributions.jl/pull/313
[#314]: https://github.com/JuliaStats/Distributions.jl/pull/314
[#321]: https://github.com/JuliaStats/Distributions.jl/pull/321
[#326]: https://github.com/JuliaStats/Distributions.jl/pull/326


