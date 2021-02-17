module FiniteVolumes

using LinearAlgebra
using StaticArrays

include("./cartesian_mesh.jl")
export CartesianMesh, PeriodicCartesianMesh, cell_centers

include("./flux_function.jl")
export LinearAdvectionFlux, Wave1DFlux, FluxFunction, ShallowWater
export directional_splitting

include("./scheme.jl")
export Upwind, NeumannBC

include("./courant.jl")

include("./euler_explicit.jl")
export FixedCourant

include("./plot_recipes.jl")

end
