
minmod      = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? min(a, b) : max(a, b))
superbee    = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? max(min(2*a, b), min(a, 2*b)) : min(max(2*a, b), max(a, 2*b)))
ultrabee(β) = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? 2*max(0, min((1/β-1)*a, b)) : -ultrabee(β)(-a, -b))

all_cells(wi) = true
no_cell(wi) = false

identity(x) = x

function muscl_reconstruction(limiter, flag=all_cells, renormalize=identity)
	function reconstruction(grid, model, w, wsupp, i_cell, i_face)
		if flag(w[i_cell])
			left_∇w = left_gradient(grid, w, i_cell)
			right_∇w = right_gradient(grid, w, i_cell)
			dx = face_center_relative_to_cell(grid, i_cell, i_face)' * rotation_matrix(grid, i_face)[:, 1]
			if dx > 0.0
				limited_∇w = limiter.(left_∇w, right_∇w)
			else
				limited_∇w = limiter.(right_∇w, left_∇w)
			end
			reconstructed_w = w[i_cell] + dx*limited_∇w
			reconstructed_w = renormalize(reconstructed_w)
			reconstructed_wsupp = compute_wsupp(model, reconstructed_w)
			return rotate_state(reconstructed_w, reconstructed_wsupp, model, rotation_matrix(grid, i_face))
		else
			return rotate_state(w[i_cell], wsupp[i_cell], model, rotation_matrix(grid, i_face))
		end
	end
end


