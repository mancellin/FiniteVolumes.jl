module FiniteVolumes

using LinearAlgebra
using StaticArrays

include("./cartesian_mesh.jl")
using FiniteVolumes.CartesianMeshes
export CartesianMesh, PeriodicCartesianMesh, cell_centers

include("./flux_function.jl")
export LinearAdvectionFlux, FluxFunction, ShallowWater
export directional_splitting

include("./scheme.jl")
export Upwind, NeumannBC

include("./diffusion.jl")
export DiffusionFlux

include("./courant.jl")

include("./euler_explicit.jl")
export FixedCourant

include("./plot_recipes.jl")

include("./experimental.jl")

end
