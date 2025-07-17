```@meta
CurrentModule = Wahoo
```
Documentation for [Wahoo](https://github.com/scheidan/Wahoo.jl).



# Wahoo.jl: Animal Geolocation with Convolution Algorithms

Wahoo is a package designed for tracking the movement of marine
animals using probabilistic state-space models.

```@raw html
<img src="assets/smoothing_animated.gif" style="width: 75%; height: auto;" />
```


# References

The filter and smoother implementation is based on:

Thygesen, Uffe Høgsbro, Martin Wæver Pedersen, and Henrik
Madsen. 2009. “Geolocating Fish Using Hidden Markov Models and Data Storage Tags.” In Tagging and Tracking of Marine Animals with Electronic Devices, 277–93. Dordrecht: Springer Netherlands. [https://doi.org/10.1007/978-1-4020-9640-2_17](https://doi.org/10.1007/978-1-4020-9640-2_17).


The example bathymetry data is derived from the following survey:

Howe JA, Anderton R, Arosio R, et al. The seabed geomorphology and geological structure of the Firth of Lorn, western Scotland, UK, as revealed by multibeam echo-sounder survey. Earth and Environmental Science Transactions of the Royal Society of Edinburgh. 2014;105(4):273-284. doi:10.1017/S1755691015000146


For an alternative, particle-based implementation see:

Lavender, E., Scheidegger, A., Albert, C., Biber, S.W., Illian, J., Thorburn, J., Smout, S., Moor, H., 2025. patter: Particle algorithms for animal tracking in R and Julia. Methods in Ecology and Evolution. [https://doi.org/10.1111/2041-210X.70029](https://doi.org/10.1111/2041-210X.70029).
