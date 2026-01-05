#include "vecfield.hpp"
#include "field.hpp"

#include "faraday.hpp"
#include "ampere.hpp"
#include "ohm.hpp"
#include "utils.hpp"
#include "gridlayout.hpp"
#include "boundary_condition.hpp"
#include "moments.hpp"
#include "pusher.hpp"
#include "diagnostics.hpp"
#include "population.hpp"

#include "highfive/highfive.hpp"

#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <memory>
#include <algorithm>




template<std::size_t dimension>
void average(Field<dimension> const& F1, Field<dimension> const& F2, Field<dimension>& Favg)
{
    // use std::transform to do an average of F1 and F2
    std::transform(F1.begin(), F1.end(), F2.begin(), Favg.begin(),
                   [](double a, double b) { return 0.5 * (a + b); });
}


template<std::size_t dimension>
void average(VecField<dimension> const& V1, VecField<dimension> const& V2,
             VecField<dimension>& Vavg)
{
    average(V1.x, V2.x, Vavg.x);
    average(V1.y, V2.y, Vavg.y);
    average(V1.z, V2.z, Vavg.z);
}


double bx(double x)
{
    // Placeholder for a function that returns Bx based on x
    return 0.0; // Example value
}

double by(double x)
{
    // Placeholder for a function that returns By based on x
    return 1.0; // Example value
}


double bz(double x)
{
    // Placeholder for a function that returns Bz based on x
    return 0.0; // Example value
}

double density(double x)
{
    // Placeholder for a function that returns density based on x
    return 1.0; // Example value
}


void magnetic_init(VecField<1>& B, GridLayout<1> const& layout)
{
    // Initialize magnetic field B
    for (auto ix = layout.primal_dom_start(Direction::X); ix <= layout.primal_dom_end(Direction::X);
         ++ix)
    {
        auto x = layout.coordinate(Direction::X, Quantity::Bx, ix);

        B.x(ix) = bx(x); // Bx
    }
    for (auto ix = layout.dual_dom_start(Direction::X); ix <= layout.dual_dom_end(Direction::X);
         ++ix)
    {
        auto x = layout.coordinate(Direction::X, Quantity::By, ix);

        B.y(ix) = by(x); // By
        B.z(ix) = bz(x); // Bz, uniform magnetic field in z-direction
    }
}




int main()
{
    double time                     = 0.;
    double final_time               = 10.0000;
    double dt                       = 0.001;
    std::size_t constexpr dimension = 1;

    std::array<std::size_t, dimension> grid_size = {100};
    std::array<double, dimension> cell_size      = {0.2};
    auto constexpr nbr_ghosts                    = 1;
    auto constexpr nppc                          = 100;

    auto layout = std::make_shared<GridLayout<dimension>>(grid_size, cell_size, nbr_ghosts);

    VecField<dimension> E{layout, {Quantity::Ex, Quantity::Ey, Quantity::Ez}};
    VecField<dimension> B{layout, {Quantity::Bx, Quantity::By, Quantity::Bz}};
    VecField<dimension> Enew{layout, {Quantity::Ex, Quantity::Ey, Quantity::Ez}};
    VecField<dimension> Bnew{layout, {Quantity::Bx, Quantity::By, Quantity::Bz}};
    VecField<dimension> Eavg{layout, {Quantity::Ex, Quantity::Ey, Quantity::Ez}};
    VecField<dimension> Bavg{layout, {Quantity::Bx, Quantity::By, Quantity::Bz}};
    VecField<dimension> J{layout, {Quantity::Jx, Quantity::Jy, Quantity::Jz}};
    VecField<dimension> V{layout, {Quantity::Vx, Quantity::Vy, Quantity::Vz}};
    Field<dimension> N{layout->allocate(Quantity::N), Quantity::N};

    auto boundary_condition = BoundaryConditionFactory<dimension>::create("periodic", layout);

    std::vector<Population<1>> populations;
    populations.emplace_back("main", layout);
    for (auto& pop : populations)
        pop.load_particles(nppc, density);


    magnetic_init(B, *layout);
    boundary_condition->fill(B);

    // Faraday<dimension> faraday{layout, dt};  // TODO uncomment when Faraday is implemented
    Faraday<dimension> faraday{layout, dt};
    Ampere<dimension> ampere{layout};
    Ohm<dimension> ohm{layout};
    Boris<dimension> push{layout, dt};



    ampere(B, J);
    boundary_condition->fill(J);
    for (auto& pop : populations)
    {
        pop.deposit();
        boundary_condition->fill(pop.flux());
        boundary_condition->fill(pop.density());
    }

    total_density(populations, N);
    bulk_velocity<dimension>(populations, N, V);
    ohm(B, J, N, V, E);
    boundary_condition->fill(E);

    diags_write_fields(B, E, V, N, time, HighFive::File::Truncate);
    diags_write_particles(populations, time, HighFive::File::Truncate);

    while (time < final_time)
    {
        std::cout << "Time: " << time << " / " << final_time << "\n";

        // TODO implement ICN temporal integration
        // Prediction–Prediction–Correction (Iterative Crank–Nicolson) 

        // Save old fields (time n)
        VecField<dimension> E0 = E;
        VecField<dimension> B0 = B;

        // Save particles at time n (ICN requires restarting from the same state)
        std::vector<std::vector<Particle<dimension>>> particles0;
        particles0.reserve(populations.size());
        for (auto& pop : populations) particles0.push_back(pop.particles());
        
        // PREDICTION 1
        // Restore particles to time n
        for (std::size_t s = 0; s < populations.size(); ++s)
            populations[s].particles() = particles0[s];

        // Field prediction: B^n+1 fom B^n, E^n
        faraday(B, E, Bnew);
        boundary_condition->fill(Bnew);
        
        // J^n+1 0 curl(B^n+1)
        ampere(Bnew, J);
        boundary_condition->fill(J);

        // Build E^n+1 from Ohm (needs moments, start from current particles)
    
        for (auto& pop : populations)
        {
            pop.deposit();
            boundary_condition->fill(pop.flux());
            boundary_condition->fill(pop.density());
        }
        total_density(populations, N);
        bulk_velocity(populations, N, V);
    
        ohm(Bnew, J, N, V, Enew);
        boundary_condition->fill(Enew);

        // Midpoint fields (n+1/2) for pusher
        average(B0, Bnew, Bavg);
        average(E0, Enew, Eavg);
        boundary_condition->fill(Bavg);
        boundary_condition->fill(Eavg);
        
        // Push particles using midpoint fields
        for (auto& pop : populations)
    {
        push(pop.particles(), Eavg, Bavg);
        boundary_condition->particles(pop.particles());
    }

        // Recompute moments after push
        for (auto& pop : populations)
        {
            pop.deposit();
            boundary_condition->fill(pop.flux());
            boundary_condition->fill(pop.density());
        }
        total_density(populations, N);
        bulk_velocity(populations, N, V);

        // Recompute E^n+1 consistently with predicted moments 
        ohm(Bnew, J, N, V, Enew);
        boundary_condition->fill(Enew);
        
        VecField<dimension> E1 = Enew;
        VecField<dimension> B1 = Bnew;

        // PREDICTION 2 
        // Restore particles again to time n 
        for (std::size_t s = 0; s < populations.size(); ++s)
        populations[s].particles() = particles0[s];

        // Midpoint fields from prediction 1 
        average(B0, B1, Bavg);
        average(E0, E1, Eavg);
        boundary_condition->fill(Bavg);
        boundary_condition->fill(Eavg);

        // Push particles
        for (auto& pop : populations)
        {
            push(pop.particles(), Eavg, Bavg);
            boundary_condition->particles(pop.particles());
        }
    
        // Deposit and moments
        for (auto& pop : populations)
        {
            pop.deposit();
            boundary_condition->fill(pop.flux());
            boundary_condition->fill(pop.density());
        }
        total_density(populations, N);
        bulk_velocity(populations, N, V);
    
        // Update fields using midpoint E
        faraday(B0, Eavg, Bnew);
        boundary_condition->fill(Bnew);
    
        ampere(Bnew, J);
        boundary_condition->fill(J);
    
        ohm(Bnew, J, N, V, Enew);
        boundary_condition->fill(Enew);
    
        VecField<dimension> E2 = Enew;
        VecField<dimension> B2 = Bnew;
        
        // CORRECTION 
        average(E0, E2, E);
        average(B0, B2, B);
        boundary_condition->fill(E);
        boundary_condition->fill(B);

        // At the end of the step, keep particles from Prediction (already advanced)
        // (they are currently the populations' particles)

        // Advance time
        time += dt;
        diags_write_fields(B, E, V, N, time);
        diags_write_particles(populations, time);
        std::cout << "**********************************\n";
    }


    return 0;
}
