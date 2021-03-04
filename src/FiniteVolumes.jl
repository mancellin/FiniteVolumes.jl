module FiniteVolumes

using LinearAlgebra
using StaticArrays
using SparseArrays

include("./cartesian_mesh.jl")
export CartesianMesh, PeriodicCartesianMesh, cell_centers

include("./flux_function.jl")
export LinearAdvectionFlux, FluxFunction, ShallowWater
export directional_splitting

include("./scheme.jl")
export Upwind, NeumannBC

include("./courant.jl")

include("./euler_explicit.jl")
export FixedCourant

include("./plot_recipes.jl")

end
