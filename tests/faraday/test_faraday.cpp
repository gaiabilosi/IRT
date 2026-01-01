#include "faraday.hpp"
#include "vecfield.hpp"
#include "gridlayout.hpp"
#include "highfive/highfive.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>

void faraday_test()
{
    std::cout << "Running Faraday test...\n";
    
    constexpr std::size_t dimension = 1;

    // Grid parameters
    std::array<std::size_t, dimension> grid_size = {1000};
    std::array<double, dimension> cell_size      = {0.1};
    constexpr std::size_t nbr_ghosts       = 1;

    // Time step for Faraday update
    double const dt = 0.01;

    auto layout = std::make_shared<GridLayout<dimension>>(grid_size, cell_size, nbr_ghosts);
    auto const dx = layout->cell_size(Direction::X);

    // B quantities: Bx (primal), By (dual), Bz (dual)
    VecField<dimension> B{layout, {Quantity::Bx, Quantity::By, Quantity::Bz}};
    VecField<dimension> Bnew{layout, {Quantity::Bx, Quantity::By, Quantity::Bz}};

    // E quantities: Ex (dual), Ey (primal), Ez (primal)
    // (Ex exists but will not be used in 1D Faraday update)
    VecField<dimension> E{layout, {Quantity::Ex, Quantity::Ey, Quantity::Ez}};

    std::vector<double> xprimal_vals;
    std::vector<double> xdual_vals;

    std::vector<double> Ey_vals, Ez_vals;
    std::vector<double> By_old_vals, Bz_old_vals;
    std::vector<double> By_new_vals, Bz_new_vals;
    std::vector<double> dBy_vals, dBz_vals;

    //----------------------------
    // STEP 1: Choose known functions
    // Ey(x) = sin(x), Ez(x) = cos(x)
    // Start from B = 0 for simplicity.
    // By^{n+1} = By^n + dt * dEz/dx, Bz^{n+1} = Bz^n - dt * dEy/dx
    // dEz/dx = -sin(x), dEy/dx =  cos(x)
    // ΔBy = -dt * sin(x), ΔBz = -dt * cos(x)

    // Primal coordinates and E initialization on primal range (including ghosts)
    for (auto ix = layout->ghost_start(Quantity::Ey, Direction::X);
         ix <= layout->ghost_end(Quantity::Ey, Direction::X); ++ix)
    {
        double const x_primal = layout->coordinate(Direction::X, Quantity::Ey, ix);

        E.y(ix) = std::sin(x_primal); // Ey primal
        E.z(ix) = std::cos(x_primal); // Ez primal
    }

    // Dual coordinate vector for plotting / comparison
    for (auto ix = layout->ghost_start(Quantity::By, Direction::X);
         ix <= layout->ghost_end(Quantity::By, Direction::X); ++ix)
    {
        double const x_dual = layout->coordinate(Direction::X, Quantity::By, ix);
        xdual_vals.push_back(x_dual);
    }

    for (auto ix = layout->ghost_start(Quantity::Ey, Direction::X);
         ix <= layout->ghost_end(Quantity::Ey, Direction::X); ++ix)
    {
        double const x_primal = layout->coordinate(Direction::X, Quantity::Ey, ix);
        xprimal_vals.push_back(x_primal);
        Ey_vals.push_back(E.y(ix));
        Ez_vals.push_back(E.z(ix));
    }

    // Initialize B (Bx primal, By/Bz dual) to 0 on their allocations
    for (auto ix = layout->ghost_start(Quantity::Bx, Direction::X);
         ix <= layout->ghost_end(Quantity::Bx, Direction::X); ++ix)
    {
        B.x(ix) = 0.0;
        Bnew.x(ix) = 0.0;
    }

    for (auto ix = layout->ghost_start(Quantity::By, Direction::X);
         ix <= layout->ghost_end(Quantity::By, Direction::X); ++ix)
    {
        B.y(ix) = 0.0;
        B.z(ix) = 0.0;
        Bnew.y(ix) = 0.0;
        Bnew.z(ix) = 0.0;
    }

    //----------------------------
    // STEP 2: Apply Faraday update: Bnew= Fraday (B, E) 
    Faraday<dimension> faraday(layout, dt);
    faraday(B, E, Bnew);

    auto const dstart = layout->dual_dom_start(Direction::X);
    auto const dend   = layout->dual_dom_end(Direction::X);

    // Store old/new By,Bz and increments on the dual domain
    for (auto ix = dstart; ix <= dend; ++ix)
    {
        By_old_vals.push_back(B.y(ix));
        Bz_old_vals.push_back(B.z(ix));
        By_new_vals.push_back(Bnew.y(ix));
        Bz_new_vals.push_back(Bnew.z(ix));
        dBy_vals.push_back(Bnew.y(ix) - B.y(ix));
        dBz_vals.push_back(Bnew.z(ix) - B.z(ix));
    }

    //----------------------------
    // STEP 3: Numerical validation (interior dual points only)
    // We skip the last dual point if the implementation assigns it as a boundary value.
    // -----------------------------------------
    double max_err_dBy = 0.0;
    double max_err_dBz = 0.0;

    for (auto ix = dstart; ix < dend; ++ix) // interior only
    {
        double const x_dual = layout->coordinate(Direction::X, Quantity::By, ix);

        double const dBy_exact = -dt * std::sin(x_dual);
        double const dBz_exact = -dt * std::cos(x_dual);

        double const dBy_num = Bnew.y(ix) - B.y(ix);
        double const dBz_num = Bnew.z(ix) - B.z(ix);

        max_err_dBy = std::max(max_err_dBy, std::abs(dBy_num - dBy_exact));
        max_err_dBz = std::max(max_err_dBz, std::abs(dBz_num - dBz_exact));
    }

    std::cout << "Faraday check (1D):\n";
    std::cout << "  max |ΔBy - ΔBy_exact| = " << max_err_dBy << "\n";
    std::cout << "  max |ΔBz - ΔBz_exact| = " << max_err_dBz << "\n";

    // First-order derivative -> expected error ~ O(dt*dx)
    double const tol = 10.0 * dt * dx;
    bool const ok = (max_err_dBy < tol) && (max_err_dBz < tol);

    std::cout << "  tolerance = " << tol << "\n";
    std::cout << (ok ? "TEST PASSED\n" : "TEST FAILED\n");

    // -----------------------------------------
    // Write diagnostics to HDF5
    // -----------------------------------------
    std::string filename = "faraday_test.h5";
    HighFive::File file(filename, HighFive::File::Truncate);

    file.createDataSet("/xprimal", xprimal_vals);
    file.createDataSet("/xdual", xdual_vals);

    file.createDataSet("/Ey", Ey_vals);
    file.createDataSet("/Ez", Ez_vals);

    file.createDataSet("/By_old", By_old_vals);
    file.createDataSet("/Bz_old", Bz_old_vals);
    file.createDataSet("/By_new", By_new_vals);
    file.createDataSet("/Bz_new", Bz_new_vals);

    file.createDataSet("/dBy", dBy_vals);
    file.createDataSet("/dBz", dBz_vals);

    std::cout << "Faraday test finished. Output written to " << filename << "\n";
}

int main()
{
    faraday_test();
    return 0;
}
 