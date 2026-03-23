#ifndef TK_H
#define TK_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cstdlib>
#include <complex>
#include <random>
#include <map>
#include "eigen-3.4.0/Eigen/Dense"
#include <random>

using cvector = std::vector<std::complex<double>>;
using cmatrix = std::vector<std::vector<std::complex<double>>>;
using vector = std::vector<double>;
using matrix = std::vector<std::vector<double>>;
using dict = std::map<std::string, double>;

double eV2au(double eV);
double nm2au(double nm);
double kV_cm2au(double kV_cm);
double s_reversed2au(double s_reversed);

class System{
private:

    /* 
    W SŁOWNIKU DO KONSTRUKTORA:
    
    N - liczba węzłów,
    a - połowa długości układu,

    m_eff - masa efektywna elektronu,
    d1 - początek potencjału "zewnętrznego",
    V1 - wartość potencjału "zewnętrznego",
    d2 - koniec potencjału "wewnętrznego",
    V2 - wartość potencjału "wewnętrznego"

    F - pole elektryczne w potencjału zmiennego w czasie
    omega - częstość potencjału zmiennego w czasie
    dt - krok czasowy

    */

    const dict parametry;

public:

    vector positions;
    vector potential;
    vector F_potential;
    cvector wavefunction;

    const double dx; // długość kroku przestrzennego

    System(const dict& parametry);
    void print_potential();
    vector F_potential_in_time(double t);
    std::pair<vector, cmatrix> matrix_method(int n_modes, int mode_to_remember);
    std::pair<cvector, cvector> Crank_Nicholson(int n_iterations);
    cmatrix Askar_Cakmak(int time_steps, int save_every_time_steps);
    void reset_wavefunction();

};

Eigen::MatrixXcd eigenize_matrix(const cmatrix& mat);
std::complex<double> dot_product(const cvector& psi1, const cvector& psi2, double dx);

#endif