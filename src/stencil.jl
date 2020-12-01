using StaticArrays
import Base.transpose
import Base.rot180
import Base.rotl90
import Base.rotr90

struct Stencil{NY, NX, T}
    data::SMatrix{NY, NX, T}
end

Stencil{N, M}(s) where {N, M} = Stencil{N, M, eltype(s)}(s)
Stencil(s::SVector) = Stencil{length(s), 1, eltype(s)}(s)
Stencil(s::AbstractMatrix) = Stencil(SMatrix{size(s)..., eltype(s)}(s...))

_offset_i(::Stencil{NY, NX}) where {NX, NY} = (NY - 1) ÷ 2 + 1
_offset_j(::Stencil{NY, NX}) where {NX, NY} = (NX - 1) ÷ 2 + 1

Base.eltype(s::Stencil{N, M, T}) where {N, M, T} = T
Base.getindex(s::Stencil, i::Int) = getindex(s.data, i + _offset_i(s), 1)
Base.getindex(s::Stencil, i::Int, j::Int) = getindex(s.data, i + _offset_i(s), j + _offset_j(s))
Base.size(::Type{Stencil{N, M, T}}) where {N, M, T} = (N, M)
Base.size(s::Stencil) = size(s.data)

stencil_radiuses(::Stencil{N, M, T}) where {N, M, T} = ((M-1) ÷ 2, (N-1) ÷ 2)

Base.map(f, s::Stencil{N, M}) where {N, M} = Stencil{N, M}(map(f, s.data))

@generated function Base.transpose(s::Stencil{N, N, T}) where {N, T}
    indices = permutedims([(i, j) for i in 1:N, j in 1:N], (2, 1))
    items = [:(s.data[$i, $j]) for (i, j) in indices]
    quote
        Stencil{$N, $N, $T}(SMatrix{$N, $N, $T}($(items...)))
    end
end

@generated function upsidedown(s::Stencil{N, N, T}) where {N, T}
    indices = [(i, N-j+1) for i in 1:N, j in 1:N]
    items = [:(s.data[$i, $j]) for (i, j) in indices]
    quote
        Stencil{$N, $N, $T}(SMatrix{$N, $N, $T}($(items...)))
    end
end

@generated function more_above(s::Stencil{N, N, T}) where {N, T}
    below = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if j < (N+1)/2]
    above = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if j > (N+1)/2]
    quote
        +($(above...)) > +($(below...))
    end
end

@generated function more_below(s::Stencil{N, N, T}) where {N, T}
    below = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if j < (N+1)/2]
    above = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if j > (N+1)/2]
    quote
        +($(above...)) < +($(below...))
    end
end

@generated function rightsideleft(s::Stencil{N, N, T}) where {N, T}
    indices = [(N-i+1, j) for i in 1:N, j in 1:N]
    items = [:(s.data[$i, $j]) for (i, j) in indices]
    quote
        Stencil{$N, $N, $T}(SMatrix{$N, $N, $T}($(items...)))
    end
end

@generated function more_on_the_right(s::Stencil{N, N, T}) where {N, T}
    right = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if i > (N+1)/2]
    left = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if i < (N+1)/2]
    quote
        +($(left...)) < +($(right...))
    end
end

@generated function more_on_the_left(s::Stencil{N, N, T}) where {N, T}
    right = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if i > (N+1)/2]
    left = [:(s.data[$i, $j]) for i in 1:N, j in 1:N if i < (N+1)/2]
    quote
        +($(left...)) > +($(right...))
    end
end

for f in [:rotr90, :rotl90, :rot180]
    eval(
         quote
             @generated function Base.$f(s::Stencil{N, N, T}) where {N, T}
                 # Defines the function $f on Stencil.
                 # At compile time, call the function from Base on an array of indices
                 # and use the indices to compile a function on Stencils that does not call the function from Base.
                 indices = $f([(i, j) for i in 1:N, j in 1:N])
                 items = [:(s.data[$i, $j]) for (i, j) in indices]
                 quote
                     Stencil{$N, $N, $T}(SMatrix{$N, $N, $T}($(items...)))
                 end
             end
         end
        )
end

more_than_half_full(s::Stencil{3, 3}) = sum(s.data)/length(s.data) > 0.5
more_than_half_empty(s::Stencil{3, 3}) = sum(s.data)/length(s.data) < 0.5
invert(s::Stencil) =  typeof(s)(1.0 .- s.data)


###################
#  RegularMesh1D  #
###################

function Stencil(grid::RegularMesh1D, i_cell)
    left_cell = i_cell == 1 ? 1 : i_cell - 1
    right_cell = i_cell == nb_cells(grid) ? nb_cells(grid) : i_cell + 1
    return Stencil(SVector{3, Int}(left_cell, i_cell, right_cell))
end

function Base.reverse(st::Stencil{3, 1})
    return Stencil(SMatrix{3, 1, eltype(st)}(st[1, 0], st[0, 0], st[-1, 0]))
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

function _right_cell(grid::AbstractRegularMesh2D, i_cell)
	if i_cell % grid.nx == 0  # Last cell at the end of a row
        if grid isa PeriodicRegularMesh2D
            return i_cell + 1 - grid.nx
        else
            return i_cell  # "Neumann" stencil
        end
	else  # General case
		return i_cell + 1
	end
end

function _left_cell(grid::AbstractRegularMesh2D, i_cell)
	if i_cell % grid.nx == 1  # First cell at the beggining of a row
        if grid isa PeriodicRegularMesh2D
            return i_cell - 1 + grid.nx
        else
            return i_cell
        end
	else
		return i_cell - 1
	end
end

_above_cell(grid::RegularMesh2D, i_cell) = i_cell + grid.nx <= nb_cells(grid) ? i_cell + grid.nx : i_cell
_below_cell(grid::RegularMesh2D, i_cell) = i_cell - grid.nx >= 1 ? i_cell - grid.nx : i_cell

_above_cell(grid::PeriodicRegularMesh2D, i_cell) = mod(i_cell+grid.nx, Base.OneTo(nb_cells(grid)))
_below_cell(grid::PeriodicRegularMesh2D, i_cell) = mod(i_cell-grid.nx, Base.OneTo(nb_cells(grid)))


function Stencil(grid::AbstractRegularMesh2D, i_cell, stencil_radius=1)
    left = _left_cell(grid, i_cell)
    right = _right_cell(grid, i_cell)
    if stencil_radius == 1
        stencil = @SMatrix [_below_cell(grid, left)   left    _above_cell(grid, left);
                            _below_cell(grid, i_cell) i_cell  _above_cell(grid, i_cell);
                            _below_cell(grid, right)  right   _above_cell(grid, right)]
        return Stencil(stencil)
    elseif stencil_radius == 2
        leftleft = _left_cell(grid, left)
        rightright = _right_cell(grid, right)
        stencil = @SMatrix [_below_cell(grid, _below_cell(grid, leftleft))   _below_cell(grid, leftleft)   leftleft   _above_cell(grid, leftleft)   _above_cell(grid, _above_cell(grid, leftleft));
                            _below_cell(grid, _below_cell(grid, left))       _below_cell(grid, left)       left       _above_cell(grid, left)       _above_cell(grid, _above_cell(grid, left));
                            _below_cell(grid, _below_cell(grid, i_cell))     _below_cell(grid, i_cell)     i_cell     _above_cell(grid, i_cell)     _above_cell(grid, _above_cell(grid, i_cell));
                            _below_cell(grid, _below_cell(grid, right))      _below_cell(grid, right)      right      _above_cell(grid, right)      _above_cell(grid, _above_cell(grid, right));
                            _below_cell(grid, _below_cell(grid, rightright)) _below_cell(grid, rightright) rightright _above_cell(grid, rightright) _above_cell(grid, _above_cell(grid, rightright))]
        return Stencil(stencil)
    else
        error("Not implemented: stencil_radius=$stencil_radius")
    end
end

"""Stencil centered in i_cell, such that i_face is on the right."""
function oriented_stencil(mesh::AbstractRegularMesh2D, i_cell, i_face, stencil_radius=1)
    st = Stencil(mesh, i_cell, stencil_radius)
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

