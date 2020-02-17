using OffsetArrays
using StaticArrays

################################################################################
#                                      1D                                      #
################################################################################

struct RegularMesh1D
    x_min::Float64
    x_max::Float64
    nb_cells::Int64
end

@inline dx(grid::RegularMesh1D) = (grid.x_max - grid.x_min)/grid.nb_cells

@inline nb_dims(grid::RegularMesh1D) = 1

@inline nb_cells(grid::RegularMesh1D) = grid.nb_cells
@inline nb_faces(grid::RegularMesh1D) = grid.nb_cells + 1

cell_center(grid::RegularMesh1D, i_cell) = SVector{1, Float64}(dx(grid)*(Float64(i_cell) - 0.5))
face_center(grid::RegularMesh1D, i_face) = SVector{1, Float64}(dx(grid)*(Float64(i_face) - 1.0))

@inline inner_faces(grid::RegularMesh1D) = Vector(2:(nb_faces(grid)-1))
@inline cells_next_to_inner_face(grid::RegularMesh1D, i_face)::Tuple{Int64, Int64} = (i_face-1, i_face)

@inline boundary_faces(grid::RegularMesh1D) = [1, nb_faces(grid)]
@inline cell_next_to_boundary_face(grid::RegularMesh1D, i_face)::Int64 = i_face == 1 ? 1 : nb_cells(grid)

@inline face_area(grid::RegularMesh1D, i_face)::Float64 = 1.0
@inline cell_volume(grid::RegularMesh1D, i_cell)::Float64 = dx(grid)

@inline rotation_matrix(grid::RegularMesh1D, i_face) = SMatrix{1, 1, Float64}(i_face == 1 ? -1.0 : 1.0)

# Indices of the cells in the 3 stencil around i_cell
function stencil(grid::RegularMesh1D, i_cell)
    left_cell = i_cell == 1 ? 1 : i_cell - 1
    right_cell = i_cell == nb_cells(grid) ? nb_cells(grid) : i_cell + 1
    stencil = SVector{3, Int}(left_cell, i_cell, right_cell)
    return OffsetArray(stencil, -1:1)
end

function centered_gradient(grid::RegularMesh1D, w, i_cell)
    w_st = w[stencil(grid, i_cell)]
    return (w_st[1] - w_st[-1])/(2dx(grid))
end

centered_gradient(grid::RegularMesh1D, w) = [centered_gradient(grid, w, i_cell) for i_cell in 1:nb_cells(grid)]

function right_gradient(grid::RegularMesh1D, w, i_cell)
    w_st = w[stencil(grid, i_cell)]
    return (w_st[1] - w_st[0])/dx(grid)
end
right_gradient(grid::RegularMesh1D, w) = [right_gradient(grid, w, i_cell) for i_cell in 1:nb_cells(grid)]

function left_gradient(grid::RegularMesh1D, w, i_cell)
    w_st = w[stencil(grid, i_cell)]
    return (w_st[0] - w_st[-1])/dx(grid)
end
left_gradient(grid::RegularMesh1D, w) = [left_gradient(grid, w, i_cell) for i_cell in 1:nb_cells(grid)]

################################################################################
#                                      2D                                      #
################################################################################

struct PeriodicRegularMesh2D
    x_min::Float64
    x_max::Float64
    nx::Int64
    y_min::Float64
    y_max::Float64
    ny::Int64
end

PeriodicRegularMesh2D(nx::Int, ny::Int) = PeriodicRegularMesh2D(0.0, 1.0, nx, 0.0, 1.0, ny)

@inline nb_dims(grid::PeriodicRegularMesh2D) = 2

@inline nb_cells(grid::PeriodicRegularMesh2D) = grid.nx * grid.ny
@inline nb_faces(grid::PeriodicRegularMesh2D) = 2 * grid.nx * grid.ny

@inline dx(grid::PeriodicRegularMesh2D) = (grid.x_max - grid.x_min)/grid.nx
@inline dy(grid::PeriodicRegularMesh2D) = (grid.y_max - grid.y_min)/grid.ny

function cell_center(grid::PeriodicRegularMesh2D, i_cell)
    SVector{2, Float64}(
                        dx(grid)*((i_cell - 1) % grid.nx + 0.5),
                        dy(grid)*((i_cell - 1) ÷ grid.nx + 0.5)
                       )
end

@inline inner_faces(grid::PeriodicRegularMesh2D) = Vector(1:(nb_faces(grid)))

@inline _is_horizontal(i_face) = i_face % 2 == 0

function cells_next_to_inner_face(grid::PeriodicRegularMesh2D, i_face)
    i_cell_1::Int64 = 0

    # Horizontal face
    if _is_horizontal(i_face) 
        i_cell_1 = i_face/2
        return (i_cell_1, (i_cell_1 + grid.nx -1) % nb_cells(grid) + 1)

    # Vertical face
    else
        i_cell_1 = (i_face + 1)/2
        if i_cell_1 % grid.nx == 0
            return (i_cell_1, i_cell_1 + 1 - grid.nx)  # Last vertical face on the right
        else
            return (i_cell_1, i_cell_1 + 1)
        end
    end
end

@inline boundary_faces(grid::PeriodicRegularMesh2D) = []
@inline cell_next_to_boundary_face(grid::PeriodicRegularMesh2D, i_face)::Int64 = 0

@inline face_area(grid::PeriodicRegularMesh2D, i_face)::Float64 = _is_horizontal(i_face) ? dx(grid) : dy(grid)

@inline cell_volume(grid::PeriodicRegularMesh2D, i_cell)::Float64 = dx(grid)*dy(grid)

function rotation_matrix(grid::PeriodicRegularMesh2D, i_face)
    if _is_horizontal(i_face)
        return SMatrix{2, 2, Float64}(0.0, -1.0, 1.0, 0.0)
    else
        return SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 1.0)
    end
end

# Indices of the cells in the 3x3 stencil around i_cell
function stencil(grid::PeriodicRegularMesh2D, i_cell)
    if i_cell % grid.nx == 0  # Last cell at the end of a row
        right_cell = i_cell + 1 - grid.nx
    else  # General case
        right_cell = i_cell + 1
    end
    if i_cell % grid.nx == 1  # First cell at the beggining of a row
        left_cell = i_cell - 1 + grid.nx
    else
        left_cell = i_cell - 1
    end
    stencil = @SMatrix [left_cell+grid.nx i_cell+grid.nx right_cell+grid.nx;
                        left_cell         i_cell         right_cell;
                        left_cell-grid.nx i_cell-grid.nx right_cell-grid.nx]
    stencil = (s -> mod(s, Base.OneTo(nb_cells(grid)))).(stencil)
    return OffsetArray(stencil, -1:1, -1:1)
end

################################################################################
#                               Only some cells                                #
################################################################################

struct FaceSplittedMesh{M}
    mesh::M
    actual_faces::Set{Int}
end

function directional_splitting(grid::PeriodicRegularMesh2D)
    horizontal_faces = Set(i for i in 1:nb_faces(grid) if _is_horizontal(i))
    grid_y = FaceSplittedMesh{PeriodicRegularMesh2D}(grid, horizontal_faces)
    vertical_faces = Set(i for i in 1:nb_faces(grid) if !(_is_horizontal(i)))
    grid_x = FaceSplittedMesh{PeriodicRegularMesh2D}(grid, vertical_faces)
    return grid_x, grid_y
end

nb_faces(mesh::FaceSplittedMesh) = length(mesh.actual_faces)
inner_faces(mesh::FaceSplittedMesh) = inner_faces(mesh.mesh) ∩ mesh.actual_faces
boundary_faces(mesh::FaceSplittedMesh) = boundary_faces(mesh.mesh) ∩ mesh.actual_faces

@inline nb_cells(mesh::FaceSplittedMesh) = nb_cells(mesh.mesh)
@inline cells_next_to_inner_face(mesh::FaceSplittedMesh, i) = cells_next_to_inner_face(mesh.mesh, i)
@inline face_area(mesh::FaceSplittedMesh, i) = face_area(mesh.mesh, i)
@inline cell_volume(mesh::FaceSplittedMesh, i) = cell_volume(mesh.mesh, i)
@inline rotation_matrix(mesh::FaceSplittedMesh, i) = rotation_matrix(mesh.mesh, i)

