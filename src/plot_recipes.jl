using RecipesBase
using ColorTypes

@recipe function plot(grid::RegularMesh1D, w, i::Int)
    seriestype := :steppre

    x = [cell_center(grid, j)[1] + dx(grid)/2 for j in 1:nb_cells(grid)]
    data = [wj[i] for wj in w]
    x, data
end

@recipe function plot(grid::PeriodicRegularMesh2D, w, i::Int)
    seriestype := :heatmap
    seriescolor --> :viridis
    aspect_ratio --> 1

    x = LinRange(grid.x_min, grid.x_max, grid.nx)
    y = LinRange(grid.y_min, grid.y_max, grid.ny)
    field = permutedims(reshape([wj[i] for wj in w], (grid.nx, grid.ny)), (2, 1))
    x, y, field
end

@recipe function plot(grid::PeriodicRegularMesh2D, w, is::Tuple{Int, Int, Int};
                      base_colors=(XYZ(1.0, 0.0, 0.2), XYZ(0.0, 1.0, 0.2), XYZ(0.0, 0.0, 0.0)))
    seriestype := :image

    x = LinRange(grid.x_min, grid.x_max, grid.nx)
    y = LinRange(grid.y_min, grid.y_max, grid.ny)
    pixels = [wj[is[1]]*base_colors[1] + wj[is[2]]*base_colors[2] + wj[is[3]]*base_colors[3] for wj in w]
    field = permutedims(reshape(pixels, (grid.nx, grid.ny)), (2, 1))
    x, y, field
end
