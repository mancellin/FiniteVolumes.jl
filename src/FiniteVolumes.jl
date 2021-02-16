module FiniteVolumes

using LinearAlgebra
using StaticArrays

include("./cartesian_mesh.jl")
export CartesianMesh, PeriodicCartesianMesh, cell_centers

include("./flux_function.jl")
export LinearAdvectionFlux, Wave1DFlux, FluxFunction

include("./scheme.jl")
export Upwind

include("./courant.jl")

include("./euler_explicit.jl")
export FixedCourant

include("./plot_recipes.jl")

end
