#include "ampere.hpp"
#include "vecfield.hpp"
#include "gridlayout.hpp"
#include "highfive/highfive.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>

void ampere_test()
{
    std::cout << "Running Ampere test...\n";
    
    constexpr std::size_t dim = 1;

    // Grid parameters
    std::array<std::size_t, dim> grid_size = {1000};
    std::array<double, dim> cell_size      = {0.1};
    constexpr std::size_t nbr_ghosts       = 1;

    auto layout = std::make_shared<GridLayout<dim>>(grid_size, cell_size, nbr_ghosts);
    auto const dx = layout->cell_size(Direction::X);

    // Vector fields: B = (Bx, By, Bz), J = (Jx, Jy, Jz)
    VecField<dim> B{layout, {Quantity::Bx, Quantity::By, Quantity::Bz}};
    VecField<dim> J{layout, {Quantity::Jx, Quantity::Jy, Quantity::Jz}};

    // Storage for diagnostic output (HDF5)
    std::vector<double> x_primal_vals;
    std::vector<double> x_dual_vals;
    std::vector<double> Bx_vals, By_vals, Bz_vals;
    std::vector<double> Jx_vals, Jy_vals, Jz_vals;

    //----------------------------
    // STEP 1: Choose known functions
    // By(x) = cos(x), Bz(x) = sin(x), dual
    // Bx unused in 1D curl, primal
    // Jy = -dBz/dx = - cos(x)
    // Jz =  dBy/dx = - sin(x)
    
    // Initialize Bx, primal-centering 
    for (auto ix = layout->ghost_start(Quantity::Bx, Direction::X);
         ix <= layout->ghost_end(Quantity::Bx, Direction::X); ++ix)
    {
        B.x(ix) = 0.0;
        Bx_vals.push_back(B.x(ix));
    }
    
    // Initialize By and Bz on dual-centering
    for (auto ix = layout->ghost_start(Quantity::By, Direction::X);
         ix <= layout->ghost_end(Quantity::By, Direction::X); ++ix)
    {
        double const x_dual = layout->coordinate(Direction::X, Quantity::By, ix);
        B.y(ix) = std::cos(x_dual);
        By_vals.push_back(B.y(ix));
        x_dual_vals.push_back(x_dual);
    }

    for (auto ix = layout->ghost_start(Quantity::Bz, Direction::X);
         ix <= layout->ghost_end(Quantity::Bz, Direction::X); ++ix)
    {
        double const x_dual = layout->coordinate(Direction::X, Quantity::Bz, ix);
        B.z(ix) = std::sin(x_dual);
        Bz_vals.push_back(B.z(ix));
    }

    // Prepare primal coordinates for plotting quantities living on primal 
    for (auto ix = layout->primal_dom_start(Direction::X);
         ix <= layout->primal_dom_end(Direction::X); ++ix)
    {
        double const x_primal = layout->coordinate(Direction::X, Quantity::Jz, ix); // primal coordinate
        x_primal_vals.push_back(x_primal);
    }
    
    //----------------------------
    // STEP 2: Apply Ampere operator
    Ampere<dim> amp{layout};
    amp(B, J);

    // Store J values
    for (auto ix = layout->dual_dom_start(Direction::X);
         ix <= layout->dual_dom_end(Direction::X); ++ix)
    {
        Jx_vals.push_back(J.x(ix)); // Jx is dual (will be 0 in 1D)
    }

    for (auto ix = layout->primal_dom_start(Direction::X);
         ix <= layout->primal_dom_end(Direction::X); ++ix)
    {
        Jy_vals.push_back(J.y(ix)); // primal
        Jz_vals.push_back(J.z(ix)); // primal
    }

    //----------------------------
    // STEP 3: Numerical validation, compare with analytic Jy/Jz on primal nodes
    // Ampere implementation uses a backward difference and sets the first primal point to a boundary value. We skip ix=start.

    auto const start = layout->primal_dom_start(Direction::X);
    auto const end   = layout->primal_dom_end(Direction::X);

    double max_err_jy = 0.0;
    double max_err_jz = 0.0;

    for (auto ix = start + 1; ix <= end; ++ix)
    {
        double const x_primal = layout->coordinate(Direction::X, Quantity::Jy, ix);

        double const jy_exact = -std::cos(x_primal);
        double const jz_exact = -std::sin(x_primal);

        max_err_jy = std::max(max_err_jy, std::abs(J.y(ix) - jy_exact));
        max_err_jz = std::max(max_err_jz, std::abs(J.z(ix) - jz_exact));
    }

    std::cout << "Ampere check (1D):\n";
    std::cout << "  max |Jy - Jy_exact| = " << max_err_jy << "\n";
    std::cout << "  max |Jz - Jz_exact| = " << max_err_jz << "\n";

    // First-order derivative -> error is O(dx). We use a loose tolerance proportional to dx.
    double const tol = 10.0 * dx;
    bool const ok = (max_err_jy < tol) && (max_err_jz < tol);

    std::cout << "  tolerance = " << tol << "\n";
    std::cout << (ok ? "TEST PASSED\n" : "TEST FAILED\n");

    {
        std::string filename = "ampere_test.h5";
        HighFive::File file(filename, HighFive::File::Truncate);
        file.createDataSet("/xprimal", x_primal_vals);
        file.createDataSet("/xdual", x_dual_vals);
        file.createDataSet("/Bx", Bx_vals);
        file.createDataSet("/By", By_vals);
        file.createDataSet("/Bz", Bz_vals);
        file.createDataSet("/Jx", Jx_vals);
        file.createDataSet("/Jy", Jy_vals);
        file.createDataSet("/Jz", Jz_vals);

        std::cout << "Ampere test finished. Output written to " << filename << "\n";  
    }
}

int main()
{
    ampere_test();
    return 0;
}
    

    