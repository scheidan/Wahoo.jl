var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Wahoo","category":"page"},{"location":"","page":"Home","title":"Home","text":"Documentation for Wahoo.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Wahoo.jl","page":"Home","title":"Wahoo.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Wahoo is a package designed for tracking the movement of marine animals using probabilistic models","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"] add git@github.com:scheidan/Wahoo.jl.git","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The following example uses example data that come with Wahoo.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Wahoo\n\nimport GeoArrays\nusing DelimitedFiles: readdlm\n\n# This is optional. If *both* packages are imported,\n# some computations use the GPU. Otherwise, the CPU is used for everything.\nimport CUDA\nimport cuDNN","category":"page"},{"location":"#1)-Bathymetry","page":"Home","title":"1) Bathymetry","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Load the bathymetry map which provides depth information for each grid cell.","category":"page"},{"location":"","page":"Home","title":"Home","text":"pathdata = joinpath(pkgdir(Wahoo), \"example_data\")\nbathymetry_map = GeoArrays.read(joinpath(pathdata, \"bathymetry_200m.tif\"))\nGeoArrays.bbox(bathymetry_map)","category":"page"},{"location":"#2)-Depth-observations","page":"Home","title":"2) Depth observations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Read the depth measurements and define the exponential likelihood model. This model implies the fish is more likely close to the seabed.","category":"page"},{"location":"","page":"Home","title":"Home","text":"# read depth signals\ndepth_signals = readdlm(joinpath(pathdata, \"depth_observations.csv\"), ',', header=true)[1][:,2]\n\n# define likelihood\nfunction p_obs_depth_exponential(signals, t, waterdepth, dist; scale=30f0)\n    signal = signals[t]\n    if signal > waterdepth         # water is too shallow\n        return zero(waterdepth)\n    else\n        # exponential\n        Z = 1 - exp(-waterdepth/scale) # normalisation due to truncation\n        exp(-(waterdepth - signal)/scale)/(scale * Z)\n    end\nend","category":"page"},{"location":"#3)-Acoustic-observations","page":"Home","title":"3) Acoustic observations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Load acoustic detection data and specify the probability model for acoustic signals given the distance.","category":"page"},{"location":"","page":"Home","title":"Home","text":"# read acoustic signals\nacoustic_signals = readdlm(joinpath(pathdata, \"acoustic_observations.csv\"), ',', header=true)[1][:,2:3]\nacoustic_signals = Int.(acoustic_signals')\nacoustic_obs = [acoustic_signals[1,:], acoustic_signals[2,:]]\n\n# read sensor positions\nmoorings = readdlm(joinpath(pathdata, \"acoustic_moorings.csv\"), ',', header=true)[1]\nacoustic_pos = tuple.(moorings[:,2], moorings[:,3])\n\n# define likelihood\nfunction p_obs_acoustic(signals, t::Int, depth::Number, distance::Number)\n    Wahoo.p_acoustic_sigmoid(signals[t], depth, distance)\nend","category":"page"},{"location":"#4)-Define-parameters","page":"Home","title":"4) Define parameters","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We define initial distribution of the fish location and configures the model parameters such as time steps, movement capabilities of the fish, and spatial resolution of the bathymetry.","category":"page"},{"location":"","page":"Home","title":"Home","text":"\n# initial values: Matrix{Float64}\np0 = zeros(size(bathymetry_map)[1], size(bathymetry_map)[2])\nidx = GeoArrays.indices(bathymetry_map, (709757.111649658, 6.26772603565296e6)) # last known location of the fish\np0[idx] = 1\n\ntsave = 1:2:720             # time steps to save\nmovement_std = 100          # standard deviation of the fish movement for one time step [m]\nspatial_resolution = 200    # spatial resolution [m]","category":"page"},{"location":"#5)-Run-inference","page":"Home","title":"5) Run inference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Finally, we run the model inference combine all observations and assumptions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"\nres = track(pos_init = p0, bathymetry = bathymetry_map,\n            tsave = tsave,\n            spatial_resolution = spatial_resolution,\n            movement_std = movement_std,\n            observations = [depth_signals, acoustic_obs...],\n            observation_models = [p_obs_depth_exponential, acoustic_obs_models...],\n            sensor_positions = [nothing, acoustic_pos...],\n            n_trajectories = 2)\n\n# Resulting probabilities\n# Array{Float32, 4}: Ny × Nx × 1 × time\nres.pos_smoother     # Prob(s_t | y_{1...T})\nres.pos_filter       # Prob(s_t | y_{1...t}), only if `save_filter = true` was used\nres.residence_dist   # 1/T Σ Prob(s_t | y_{1...T})\nres.trajectories     # Vector of trajectories sampled from Prob(s_{1...T} | y_{1...T})\nres.log_p            # Prob(y_t)\nres.tsave            # time points","category":"page"},{"location":"#Defining-observation-models","page":"Home","title":"Defining observation models","text":"","category":"section"},{"location":"#Using-GPU","page":"Home","title":"Using GPU","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"todo","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"track","category":"page"},{"location":"#Wahoo.track","page":"Home","title":"Wahoo.track","text":"Tracks the location of the fish\n\ntrack(;pos_init::Matrix,\n       tsave::AbstractVector = 1:100,\n       bathymetry::GeoArrays.GeoArray,\n       observations::Vector,\n       observation_models::Vector{Function},\n       sensor_positions::Vector,\n       spatial_resolution,\n       movement_std,\n       save_filter::Bool = false,\n       n_trajectories::Int = 0,\n       show_progressbar::Bool = !is_logging(stderr),\n       precision = Float32)\n\nInfers the location of the animal based on a diffusion model and smoothing.\n\nKeyword Arguments\n\npos_init::Matrix: Initial probability distribution of the fish position\ntsave::AbstractVector: Time steps at which the probability map is saved.\nbathymetry: Bathymetric data as GeoArray\nspatial_resolution: the spatial resolution [m] of the bathymetry.\nmovement_std: Standard deviation of the fish movement within one time step [m]\nobservations: Vector holding all observations. Each element contains the observation of a separate sensor.\nobservation_models::Vector{Function}: Vector containing the observation model for each sensor.\nsensor_positions: Vector of tuples of coordinates or nothing, i.e. Vector{Union{Nothing, Tuple{Real, Real}}}\nsave_filter: if true the probabilities from the filter run are returned.\nn_trajectories=0: Number of trajectories to sample\nshow_progressbar = !is_logging(stderr): defaults to true for interactive use.\nprecision = Float32: numerical floating point type used for computations\n\nNote, the elements of the vectors observations, observation_models, and sensor_positions must be sorted in the same way, i.e. the elements at the same position in the Vectors refer to the same sensor.\n\nReturn\n\nA named tuple with the following elements:\n\npos_smoother: Smoothed probability distribution of the fish positions for all timesteps in tsave.\nresidence_dist: Residence distribution.\ntrajectories: Sampled trajectories if n_trajectories > 0, otherwise nothing.\nlog_p: Log probability of the observations.\ntsave: Vector of time steps at which the results are saved.\npos_filter: Filtered  probability distribution of the fish positions, included if save_filter = true.\n\n\n\n\n\n","category":"function"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The filter and smoother implementation is based on:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Thygesen, Uffe Høgsbro, Martin Wæver Pedersen, and Henrik Madsen. 2009. “Geolocating Fish Using Hidden Markov Models and Data Storage Tags.” In Tagging and Tracking of Marine Animals with Electronic Devices, 277–93. Dordrecht: Springer Netherlands. https://doi.org/10.1007/978-1-4020-9640-2_17.","category":"page"}]
}
