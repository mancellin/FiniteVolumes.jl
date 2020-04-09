module FiniteVolumes

using StaticArrays

include("./models/scalar_linear_advection.jl")
export ScalarLinearAdvection

include("./models/n_scalars_linear_advection.jl")
export NScalarLinearAdvection

include("./models/isothermal_two_fluid.jl")
export IsothermalTwoFluidEuler

include("./mesh.jl")
export RegularMesh1D, PeriodicRegularMesh2D, directional_splitting, nb_dims, nb_cells, cell_center

include("./reconstruction.jl")

include("./scheme.jl")

include("./plot_recipes.jl")

end
