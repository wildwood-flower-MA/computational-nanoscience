#ifndef HF_H
#define HF_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cstdlib>
#include <random>

#define PI 3.141592653589793
using vector = std::vector<double>;
using matrix = std::vector<std::vector<double>>;

struct parametry{

    // PARAMETRY UKŁADU
    double dx = 1.0;

    // PARAMETRY ELEKTRONU
    double m = 1.0; // masa efekywna elektronu
    double kappa = 1.0; // stała przy potencjale elektrycznym

    // PARAMETRY KROPKI KWANTOWEJ
    double a = 1.0; // długość QD
    double l = 1.0; // długość drutu, w którym znajduje się QD

    // PARAMETRY SYMULACJI
    int N = 1000000000; // maksymalna liczba relaksacji
    double dtau = 1.0; // krok w czasie urojonym relaksacji
    double tol; // tolerancja względnej różnicy energii

    };

double eV2au(double);
double nm2au(double);
double V(const vector&, const parametry&);

matrix operator+(const matrix&, const matrix&);
matrix operator-(const matrix&, const matrix&);
matrix operator*(double, const matrix&);

vector operator+(const vector&, const vector&);
vector operator-(const vector&, const vector&);
vector operator*(double, const vector&);

matrix random_matrix_NxN(int);
vector random_vector_N(int);
vector idx2xy(int, int, const parametry&);
double idx2xy(int, const parametry&);
inline double random_uniform_m11();
void save_matrix(matrix&, parametry&);
void save_vector(vector&, parametry&);

matrix Hamiltonian(const matrix&, const parametry&);
double calculate_energy(const matrix&, const parametry&);
vector relax_in_imaginary_time(matrix&, const parametry&);

vector HF_h(const vector&, const parametry&);
vector HF_J(const vector&, const parametry&);
vector HF_F_operator(const vector&, const vector&, const parametry&);
double HF_calculate_energy(const vector&,const vector&, const parametry&);
vector HF_relax_in_imaginary_time(vector&, const parametry&);

#endif
