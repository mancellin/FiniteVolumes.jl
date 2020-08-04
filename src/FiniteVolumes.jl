module FiniteVolumes

using LinearAlgebra
using StaticArrays

include("./models/abstract.jl")
export directional_splitting

include("./models/scalar_linear_advection.jl")
export ScalarLinearAdvection

include("./models/anonymous_model.jl")

include("./models/isothermal_two_fluid.jl")
export IsothermalTwoFluidEuler, full_state

include("./mesh.jl")
export RegularMesh1D, RegularMesh2D, PeriodicRegularMesh2D, nb_dims, nb_cells, cell_center

include("./stencil.jl")
export Stencil

include("./scheme.jl")
export NumericalFlux, Upwind

include("./reconstruction.jl")
export Muscl, minmod, superbee, ultrabee, all_cells, no_cell
export VOF, LagoutiereDownwind
export Hybrid

include("./plot_recipes.jl")

end
