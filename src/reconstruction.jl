
function least_square_gradient(data)
    @assert nb_dim(grid) == 2
    A = zeros((2, 2))
    b = zeros(2)
    for i_face in faces_next_to_cell(grid, i_cell)
        if !is_boundary_face(grid, i_face)
            # Not computed with i_other_cell because of periodic meshes...
            if i_cell == cells_next_to_face(grid, i_face)[1]
                dx, dy = -vector_between_cell_centers_across_face(grid, i_face)
            else
                dx, dy = vector_between_cell_centers_across_face(grid, i_face)
            end
            i_other_cell = cell_on_the_other_side_of_the_face(grid, i_cell, i_face)
            du = u[i_cell] - u[i_other_cell]
            A += [dx^2  dx*dy; dx*dy  dy^2]
            b += [du*dx, du*dy]
        end
    end

    if A[2, 2] == 0.0
        #= @warning "2D mesh that is actually 1D" =#
        return [b[1] / A[1, 1], 0.0]
    else
        return A \ b
    end
end
