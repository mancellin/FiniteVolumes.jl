module FiniteVolumes

import Base.print
import Base.show

using LinearAlgebra
using StaticArrays

include("./models/abstract.jl")
export directional_splitting

include("./models/scalar_linear_advection.jl")
export ScalarLinearAdvection

include("./models/anonymous_model.jl")

include("./models/shallow_waters.jl")
export ShallowWater

include("./models/isothermal_two_fluid.jl")
export IsothermalTwoFluidEuler, full_state

include("./mesh.jl")
export RegularMesh1D, RegularMesh2D, PeriodicRegularMesh2D, nb_dims, nb_cells, all_cells, cell_centers

include("./scheme.jl")
export NumericalFlux, Upwind, FixedCourant

include("./plot_recipes.jl")

end
