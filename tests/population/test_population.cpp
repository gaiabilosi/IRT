#include "population.hpp"
#include "moments.hpp"
#include "highfive/highfive.hpp"
#include "field.hpp"
#include "vecfield.hpp"
#include "gridlayout.hpp"

#include <iostream>
#include <vector>

// Simple uniform density function for testing
double density_profile(double x)
{
    // Constant density for simplicity
    return 1.0;
}

void test_population()
{
    std::cout << "Running population test..." << std::endl;

    constexpr std::size_t dimension = 1;

    // Step 1: Grid setup
    std::array<std::size_t, dimension> grid_size = {50};
    std::array<double, dimension> cell_size = {0.1};
    constexpr std::size_t nbr_ghosts = 1;
    constexpr std::size_t nppc = 5000; // number of particles per cell

    auto layout = std::make_shared<GridLayout<dimension>>(grid_size, cell_size, nbr_ghosts);

    // Step 2: Create and load population
    Population<dimension> pop("main", layout);
    pop.load_particles(nppc, density_profile);
    pop.deposit(); // deposit charge and flux on the grid

    // Check: total deposited density equals total particle weight (mass conservation in deposit) 
    double sumW = 0.0;
    for (auto const& p : pop.particles())
        sumW += p.weight;

    double sumNpop = 0.0;
    for (std::size_t i = 0; i < pop.density().data().size(); ++i)
        sumNpop += pop.density()(i);

    std::cout << "Check deposit conservation:\n";
    std::cout << "  sum(weights)      = " << sumW << "\n";
    std::cout << "  sum(pop.density)  = " << sumNpop << "\n";
    std::cout << "  difference        = " << (sumNpop - sumW) << "\n";

    // Step 3: Wrap in vector for moments functions
    std::vector<Population<dimension>> populations = {pop};

    // Step 4: Create total density and velocity fields
    Field<dimension> N{layout->allocate(Quantity::N), Quantity::N};
    VecField<dimension> V{layout, {Quantity::Vx, Quantity::Vy, Quantity::Vz}};

    // Step 5: Compute total density and bulk velocity 
    total_density(populations, N);
    bulk_velocity(populations, N, V);

    // Check: global mean Vx on the grid should be close to 0 (Maxwellian with mean 0)
    double meanVx = 0.0;
    std::size_t count = 0;
    
    for (std::size_t i = 0; i < V.x.data().size(); ++i)
    {
        meanVx += V.x(i);
        ++count;
    }
    
    meanVx /= static_cast<double>(count);
    
    std::cout << "Check bulk velocity:\n";
    std::cout << "  mean(Vx) over grid = " << meanVx << " (should be close to 0)\n";

    // Step 6: Simple print to check
    std::cout << "First 5 density values: ";
    for (int i = 0; i < 5; ++i) std::cout << N(i) << " ";
    std::cout << "\n";

    std::cout << "First 5 Vx values: ";
    for (int i = 0; i < 5; ++i) std::cout << V.x(i) << " ";
    std::cout << "\n";

    // Step 7: Output results to check
    std::string filename = "test_population_output.h5";
    HighFive::File file(filename, HighFive::File::Truncate);
    file.createDataSet("/density", N.data());
    file.createDataSet("/vx", V.x.data());
    file.createDataSet("/vy", V.y.data());
    file.createDataSet("/vz", V.z.data());

    std::cout << "Results written to " << filename << std::endl;
}

int main()
{
    test_population();
    return 0;
}