#ifndef HF_H
#define HF_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cstdlib>
#include <random>
#include "eigen-3.4.0/Eigen/Dense"

#define PI 3.141592653589793
using vector = std::vector<double>;
using matrix = std::vector<std::vector<double>>;

struct parametry{

    // PARAMETRY UKŁADU
    std::size_t n = 9; // układ ma n^2 węzłów 
    double dx = 1.0; // odległości międzywęzłowe

    // PARAMETRY ELEKTRONU
    double m = 1.0; // masa efekywna elektronu
    double E = 1.0; // energia elektronu
    double alpha_x = 1.0; // wariancja gaussianu w x
    double alpha_y = 1.0; // wariancja gaussianu w y

    // PARAMETRY KROPKI KWANTOWEJ
    double omega_x = 1.0; // parametr skalujący w x
    double omega_y = 1.0; // parametr skalujący w y
    };

double eV2au(double);
double nm2au(double);
std::vector<double> idx2xy(int, const parametry&);
std::vector<std::vector<double>> S_matrix(const parametry&);
std::vector<std::vector<double>> K_matrix(const parametry&);
std::vector<std::vector<double>> V_matrix(const parametry&);
Eigen::MatrixXd eigenize_matrix(const matrix&);
std::pair<vector, matrix> Hc_ESc(matrix&, matrix&, parametry&);
vector Hc_ESc_energies(matrix&, matrix&, parametry&);

void zapisz(std::ostream&, const matrix&, const parametry&);
void plot_gaussian(int, const parametry&);
matrix make_gaussian(int, const parametry&);
void plot_psi(matrix&, int, parametry&);
void print_matrix(matrix&);
void plot_3lowest(parametry&, double, double, double);
vector n_lowest_analytical(int, parametry&);
matrix operator+(const matrix&, const matrix&);
matrix operator-(const matrix&, const matrix&);
matrix operator*(double, const matrix&);

// UPDATE 25.03
matrix make_gaussian_v2(int, const parametry&);
void plot_psi_v2(matrix&, int, parametry&);
void plot_gaussian_v2(int, parametry&);
vector n_lowest_just_x(int, double);

#endif
