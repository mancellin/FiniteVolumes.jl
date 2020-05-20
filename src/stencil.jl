import Base.getindex
import Base.size
import Base.rotr90
import Base.rotl90
import Base.rot180
import Base.transpose
import Base.reverse

struct Stencil{NY, NX, T}
    data::SMatrix{NY, NX, T}
end

Stencil(s::SVector) = Stencil{length(s), 1, eltype(s)}(s)
Stencil(s::AbstractMatrix) = Stencil(SMatrix{size(s)..., eltype(s)}(s...))

offset_i(::Stencil{NY, NX, T}) where {NX, NY, T} = (NY - 1) ÷ 2 + 1
offset_j(::Stencil{NY, NX, T}) where {NX, NY, T} = (NX - 1) ÷ 2 + 1

getindex(s::Stencil, i::Int) = getindex(s.data, i + offset_i(s), 1)
getindex(s::Stencil, i::Int, j::Int) = getindex(s.data, i + offset_i(s), j + offset_j(s))
size(s::Stencil) = size(s.data)

function reverse(st::Stencil{3, 1, T}) where T
    return Stencil(SMatrix{3, 1, T}(st[1, 0], st[0, 0], st[-1, 0]))
end

function rotr90(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[1, -1]  st[0, -1]  st[-1, -1];
                                      st[1, 0]   st[0, 0]   st[-1, 0];
                                      st[1, 1]   st[0, 1]   st[-1, 1]]
                           )
end

function rotl90(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[-1, 1]  st[0, 1]  st[1, 1];
                                      st[-1, 0]  st[0, 0]  st[1, 0];
                                      st[-1, -1] st[0, -1] st[1, -1]]
                           )
end

function rot180(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[1, 1]   st[1, 0]   st[1, -1];
                                      st[0, 1]   st[0, 0]   st[0, -1];
                                      st[-1, 1]  st[-1, 0]  st[-1, -1]]
                           )
end

function transpose(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                       @SMatrix [st[-1, -1]  st[0, -1]  st[1, -1];
                                 st[-1, 0]   st[0, 0]   st[1, 0];
                                 st[-1, 1]   st[0, 1]   st[1, 1]]
                      )
end

more_above(s::Stencil{3, 3, T}) where T = s[-1, 1] + s[0, 1] + s[1, 1] > s[-1, -1] + s[0, -1] + s[1, -1]
more_below(s::Stencil{3, 3, T}) where T = s[-1, 1] + s[0, 1] + s[1, 1] < s[-1, -1] + s[0, -1] + s[1, -1]

function upsidedown(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[-1, 1]  st[-1, 0]  st[-1, -1];
                                      st[0, 1]   st[0, 0]   st[0, -1];
                                      st[1, 1]   st[1, 0]   st[1, -1]]
                           )
end

more_on_the_left(s::Stencil{3, 3, T}) where T = s[-1, -1] + s[-1, 0] + s[-1, 1] > s[1, -1] + s[1, 0] + s[1, 1]
more_on_the_right(s::Stencil{3, 3, T}) where T = s[-1, -1] + s[-1, 0] + s[-1, 1] < s[1, -1] + s[1, 0] + s[1, 1]

function rightsideleft(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[1, -1]  st[1, 0]  st[1, 1];
                                      st[0, -1]  st[0, 0]  st[0, 1];
                                      st[-1, -1] st[-1, 0] st[-1, 1]]
                           )
end

more_than_half_full(s::Stencil{3, 3, T}) where T = sum(s.data)/length(s.data) > 0.5

###################
#  RegularMesh1D  #
###################

function Stencil(grid::RegularMesh1D, i_cell)
    left_cell = i_cell == 1 ? 1 : i_cell - 1
    right_cell = i_cell == nb_cells(grid) ? nb_cells(grid) : i_cell + 1
    return Stencil(SVector{3, Int}(left_cell, i_cell, right_cell))
end

function oriented_stencil(mesh::RegularMesh1D, i_cell, i_face)
    st = Stencil(mesh, i_cell)
    if face_center_relative_to_cell(mesh, i_cell, i_face)' * rotation_matrix(mesh, i_face)[:, 1] > 0
        return st
    else
        return reverse(st)
    end
end


###################
#  RegularMesh2D  #
###################

function Stencil(grid::AbstractRegularMesh2D, i_cell)
	if i_cell % grid.nx == 0  # Last cell at the end of a row
        if grid isa PeriodicRegularMesh2D
            right_cell = i_cell + 1 - grid.nx
        else
            right_cell = i_cell  # "Neumann" stencil
        end
	else  # General case
		right_cell = i_cell + 1
	end

	if i_cell % grid.nx == 1  # First cell at the beggining of a row
        if grid isa PeriodicRegularMesh2D
            left_cell = i_cell - 1 + grid.nx
        else
            left_cell = i_cell
        end
	else
		left_cell = i_cell - 1
	end

	stencil = @SMatrix [left_cell-grid.nx   left_cell    left_cell+grid.nx;
						i_cell-grid.nx      i_cell       i_cell+grid.nx;
                        right_cell-grid.nx  right_cell   right_cell+grid.nx]

    if grid isa PeriodicRegularMesh2D
	    stencil = (s -> mod(s, Base.OneTo(nb_cells(grid)))).(stencil)
    else
        stencil = (s -> s < 1 ? s + grid.nx : (s > nb_cells(grid) ? s - grid.nx : s)).(stencil)
    end

    return Stencil(stencil)
end

function oriented_stencil(mesh::AbstractRegularMesh2D, i_cell, i_face)
    st = Stencil(mesh, i_cell)
    if _is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[1]
        st = rotr90(st)
    elseif _is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[2]
        st = rotl90(st)
    elseif !_is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[2]
        st = rot180(st) 
    end
    return st
end


###########################
#  To put somewhere else  #
###########################

function central_differences_gradient(grid::PeriodicRegularMesh2D, w, i_cell)
	st = Stencil(grid, i_cell)
	dwdx = (w[st[1, 0]] - w[st[-1, 0]])/(2dx(grid))
	dwdy = (w[st[0, 1]] - w[st[0, -1]])/(2dy(grid))
	return SVector{2, Float64}(dwdx, dwdy)
end

function youngs_gradient(grid::PeriodicRegularMesh2D, w, i_cell)
	st = Stencil(grid, i_cell)
	dwdx = (w[st[1, 1]] + 2w[st[1, 0]] + w[st[1, -1]] - w[st[-1, 1]] - 2w[st[-1, 0]] - w[st[-1, -1]])/(8dx(grid))
	dwdy = (w[st[1, 1]] + 2w[st[0, 1]] + w[st[-1, 1]] - w[st[1, -1]] - 2w[st[0, -1]] - w[st[-1, -1]])/(8dy(grid))
	return SVector{2, Float64}(dwdx, dwdy)
end

function least_square_gradient(grid::PeriodicRegularMesh2D, w, i_cell)
	st = Stencil(grid, i_cell)
    A = @SMatrix zeros(2, 2)
    b = @SVector zeros(2)
	for i in -1:1, j in -1:1
		Δx = i*dx(grid)
		Δy = j*dy(grid)
		Δu = w[st[i, j]] - w[st[0, 0]]
		A = A + SMatrix{2, 2, Float64}(Δx^2, Δx*Δy, Δx*Δy, Δy^2)
		b = b + SVector{2, Float64}(Δu*Δx, Δu*Δy)
	end

    if A[2, 2] == 0.0
        #= @warning "2D mesh that is actually 1D" =#
        return [b[1] / A[1, 1], 0.0]
    else
        return A \ b
    end
end

