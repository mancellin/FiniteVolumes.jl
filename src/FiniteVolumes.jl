module FiniteVolumes

using StaticArrays

include("./models/abstract.jl")

include("./models/scalar_linear_advection.jl")
export ScalarLinearAdvection

include("./models/isothermal_two_fluid.jl")
export IsothermalTwoFluidEuler

include("./mesh.jl")
export RegularMesh1D, PeriodicRegularMesh2D, directional_splitting, nb_dims, nb_cells, cell_center

include("./scheme.jl")
export Upwind

include("./reconstruction.jl")
export Muscl, minmod, superbee, ultrabee, all_cells, no_cell
export VOF, LagoutiereDownwind

include("./plot_recipes.jl")

end
