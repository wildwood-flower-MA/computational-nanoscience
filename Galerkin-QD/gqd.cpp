#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cstdlib>
#include <random>
#include "eigen-3.4.0/Eigen/Dense"
#include "gqd.h"

#define PI 3.141592653589793
using vector = std::vector<double>;
using matrix = std::vector<std::vector<double>>;

// funkcja do konwersji eV -> a.u.
double eV2au(double eV){
    return 0.036749325871*eV;
}

// funkcja do konwersji nm -> a.u.
double nm2au(double nm){
    return 18.897261339*nm;
}

vector idx2xy(int idx, const parametry& pm){
    // indeks -> położenie węzła o zadanym
    // indeksie w układzie współrzędnych
    double a = 0.5*pm.dx*(pm.n - 1);
    int i = (int)(idx/pm.n);
    int j = (int)(idx%pm.n);

    return {-a+i*pm.dx, -a+j*pm.dx};
}

double gaussian(double x, double y, int idx, const parametry& pm){
    // funkcja licząca wartość gaussianu w punkcie (x,y)
    // skoncentrowanego wokół punktu (x[idx], y[idx])
    // o parametrze skalującym określonym w parametrach pm

    vector pos = idx2xy(idx, pm);
    double constant = 1.0/pow(PI*PI*pm.alpha_x*pm.alpha_y, 0.25);

    return constant*std::exp(-0.5*pow(x-pos[0], 2.0)/pm.alpha_x)*std::exp(-0.5*pow(y-pos[1], 2.0)/pm.alpha_y);
}

matrix operator+(const matrix& m1, const matrix& m2){

    size_t size1 = m1.size();
    size_t size2 = m2.size();
    if(size1 != size2){
        throw std::runtime_error("kurcze chlopaki cos nie tak z macierzami");
    }
    matrix temp = matrix(size1, vector(size1));
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < size1; j++){
            temp[i][j] = m1[i][j]+m2[i][j];
        }
    }
    return temp;
}

matrix operator-(const matrix& m1, const matrix& m2){

    size_t size1 = m1.size();
    size_t size2 = m2.size();
    if(size1 != size2){
        throw std::runtime_error("kurcze chlopaki cos nie tak z macierzami");
    }
    matrix temp = matrix(size1, vector(size1));
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < size1; j++){
            temp[i][j] = m1[i][j]-m2[i][j];
        }
    }
    return temp;
}

matrix operator*(double constant, const matrix& mat){

    size_t size = mat.size();
    matrix temp = matrix(size, vector(size));

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            temp[i][j] = constant*mat[i][j];
        }
    }
    return temp;
}

matrix S_matrix(const parametry& pm){
    // znajdowanie elementów macierzy przekrywania funkcji bazowych, czyli S,
    // S_ij = <gaussian_i | gaussian_j>

    auto s_matrix = matrix(pm.n*pm.n, vector(pm.n*pm.n));
    vector pos_i, pos_j;

    for(int i = 0; i<pm.n*pm.n; i++){
        for(int j = 0; j<pm.n*pm.n; j++){
            
            // znajduję położenia węzłów o zadanych indeksach
            pos_i = idx2xy(i, pm);
            pos_j = idx2xy(j, pm);
            
            // liczę elementy macierzy S
            s_matrix[i][j] = std::exp(-pow(pos_i[0]-pos_j[0], 2.0)/pm.alpha_x/4.0
                                - pow(pos_i[1]-pos_j[1], 2.0)/pm.alpha_y/4.0);

        }
    }

    return s_matrix;
}

matrix K_matrix(const parametry& pm){
    // znajdowanie elementów macierzy energii kinetycznej, czyli K,
    // K_ij = -1/2m <gaussian_i | partial^2_x + partial^2_y | gaussian_j>

    auto k_matrix = matrix(pm.n*pm.n, vector(pm.n*pm.n));
    double delta2_x, delta2_y, s_element;
    vector pos_i, pos_j;

    for(int i = 0; i<pm.n*pm.n; i++){
        for(int j = 0; j<pm.n*pm.n; j++){

            // znajduję położenia węzłów o zadanych indeksach
            pos_i = idx2xy(i, pm);
            pos_j = idx2xy(j, pm);
            
            // znajduję kwadraty różnic ich współrzędnych położeniowych
            delta2_x = pow(pos_i[0]-pos_j[0], 2.0);
            delta2_y = pow(pos_i[1]-pos_j[1], 2.0);

            // wyznaczam element macierzy S,
            // poprzez który wyrażony jest wzór na element macierzy K
            s_element = std::exp(-delta2_x/pm.alpha_x/4.0
                                - delta2_y/pm.alpha_y/4.0);

            // liczę elementy macierzy V
            k_matrix[i][j] = -0.5*s_element*((delta2_x-2.0*pm.alpha_x)/(pm.alpha_x*pm.alpha_x)/4.0
                                        + (delta2_y-2.0*pm.alpha_y)/(pm.alpha_y*pm.alpha_y)/4.0)/pm.m;
        }
    }

    return k_matrix;
}

matrix V_matrix(const parametry& pm){
    // znajdowanie elementów macierzy energii potencjalnej, czyli V,
    // V_ij = 1/2 m<gaussian_i | omega_x x^2 + omega_y y^2 | gaussian_j>

    auto v_matrix = matrix(pm.n*pm.n, vector(pm.n*pm.n));
    double delta2_x, delta2_y, s_element, adelta2_x, adelta2_y;
    vector pos_i, pos_j;

    for(int i = 0; i<pm.n*pm.n; i++){
        for(int j = 0; j<pm.n*pm.n; j++){

            // znajduję położenia węzłów o zadanych indeksach
            pos_i = idx2xy(i, pm);
            pos_j = idx2xy(j, pm);

            // znajduję kwadraty różnic ich współrzędnych położeniowych
            // oraz kwadraty ich sum
            delta2_x = pow(pos_i[0]-pos_j[0], 2.0);
            delta2_y = pow(pos_i[1]-pos_j[1], 2.0);
            adelta2_x = pow(pos_i[0]+pos_j[0], 2.0);
            adelta2_y = pow(pos_i[1]+pos_j[1], 2.0);

            // wyznaczam element macierzy S,
            // poprzez który wyrażony jest wzór na element macierzy V
            s_element = std::exp(-delta2_x/pm.alpha_x/4.0
                                - delta2_y/pm.alpha_y/4.0);

            // liczę elementy macierzy V
            v_matrix[i][j] = 0.5*pm.m*s_element*(pm.omega_x*pm.omega_x*
                                                (adelta2_x+2.0*pm.alpha_x)/4.0
                                                + pm.omega_y*pm.omega_y*
                                                (adelta2_y+2.0*pm.alpha_y)/4.0);
        }
    }

    return v_matrix;
}

Eigen::MatrixXd eigenize_matrix(const matrix& mat){

    int size = mat.size();
    Eigen::MatrixXd matrix(size, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix(i, j) = mat[i][j];
        }
    }
    return matrix;
}

std::pair<vector, matrix> Hc_ESc(matrix& H, matrix& S, parametry& pm){

    Eigen::MatrixXd H_eigen = eigenize_matrix(H);
    Eigen::MatrixXd S_eigen = eigenize_matrix(S);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_eigen, S_eigen);

    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    vector E(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());

    Eigen::MatrixXd eigenvectors = solver.eigenvectors();
    matrix v;
    for (int col = 0; col < eigenvectors.cols(); col++) {
        vector column(eigenvectors.rows());
        for (int row = 0; row < eigenvectors.rows(); row++) {
            column[row] = eigenvectors(row, col);
        }
        v.push_back(column);
    }

    return {E, v};
}

vector Hc_ESc_energies(matrix& H, matrix& S, parametry& pm){

    Eigen::MatrixXd H_eigen = eigenize_matrix(H);
    Eigen::MatrixXd S_eigen = eigenize_matrix(S);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_eigen, S_eigen);

    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    vector E(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());

    return E;
}

void zapisz(std::ostream& my_out, const matrix& vec, const parametry& pm){

    my_out << pm.n << " " << pm.dx*(pm.n-1.0)/2.0/nm2au(1.0) << " " << pm.dx  << " " << pm.alpha_x << " " << pm.alpha_y << "\n";
    for (const auto& row : vec) {
        for (const auto& element : row) {
            my_out << element << " ";
        }
        my_out << "\n";
    }
    return;
}


void plot_gaussian_v2(int idx, parametry& pm){

    matrix psi = make_gaussian_v2(idx, pm);

    std::ofstream plik("temp.dat");
    if(!plik.is_open()){ return; }
    zapisz(plik, psi, pm);
    plik.close();
    std::string command = "python3 gqd_psi.py";
    int wynik = system(command.c_str());
    remove("temp.dat");
    
    return;
}


matrix make_gaussian_v2(int idx, const parametry& pm){
    matrix the_gaussian = matrix(100, vector(100));
    double x, y;

    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            x = -0.5*pm.dx*(pm.n - 1) + i*pm.dx*(pm.n - 1)/99; // przeindeksowanie
            y = -0.5*pm.dx*(pm.n - 1) + j*pm.dx*(pm.n - 1)/99;
            the_gaussian[i][j] = gaussian(x, y, idx, pm);
        }
    }

    return the_gaussian;
}

void plot_psi_v2(matrix& c, int mode, parametry& pm){

    matrix psi = matrix(100, vector(100, 0.0));

    for(int k = 0; k < pm.n * pm.n; k++){
        matrix psi_n = c[mode][k]*make_gaussian_v2(k, pm);
        psi = psi + psi_n;
    }

    matrix psi2 = matrix(100, vector(100));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            psi2[i][j] = psi[i][j]*psi[i][j];
        }
    }

    std::ofstream plik("temp.dat");
    if(!plik.is_open()){ return; }
    zapisz(plik, psi2, pm);
    plik.close();
    std::string command = "python3 gqd_psi.py";
    int wynik = system(command.c_str());
    remove("temp.dat");
    
    return;
}


void print_matrix(matrix& mat){
    for (vector row : mat){
        for (double element : row){
            std::cout << element << " ";
        }
        std::cout << "\n";
    }
}

vector n_lowest_analytical(int n, parametry& pm){
    vector energies, energies_n;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            energies.push_back(pm.omega_x*(i+0.5) + pm.omega_y*(j+0.5));
        }
    }
    std::sort(energies.begin(), energies.end());
    for( int i = 0; i < n; i++){
        energies_n.push_back(energies[i]);
    }
    return energies_n;
}

vector n_lowest_just_x(int n, double omega){
    vector energies, energies_n;

    for(int i = 0; i < n; i++){
        energies.push_back(omega*(i+0.5));
    }
    std::sort(energies.begin(), energies.end());
    for( int i = 0; i < n; i++){
        energies_n.push_back(energies[i]);
    }
    return energies_n;
}

void plot_10lowest(parametry& pm, double omega_x_min,
    double omega_x_max, double omega_x_step){

    matrix S, K, V, E;
    vector energies, n_lowest;
    std::ofstream plik("temp.dat");
    if(!plik.is_open()){ return; }

    for(double omega_x = omega_x_min; omega_x <= omega_x_max;
        omega_x += omega_x_step){

        pm.omega_x = omega_x;
        S = S_matrix(pm);
        K = K_matrix(pm);
        V = V_matrix(pm);
        E = K+V;

        n_lowest = n_lowest_analytical(10, pm);
        
        plik << 1000.0*omega_x/eV2au(1.0) << " ";
        energies = Hc_ESc_energies(E, S, pm);
        for(int i  = 0; i<10; i++){
            plik << 1000.0*energies[i]/eV2au(1.0) << " ";
        }

        for(int i  = 0; i<10; i++){
            plik << 1000.0*n_lowest[i]/eV2au(1.0) << " ";
        }
        plik << "\n";
    }

    plik.close();
    std::string command = "python3 gqd_energy.py";
    int wynik = system(command.c_str());
    remove("temp.dat");

}

// ---------------------------------- (MAIN) ----------

int main(){

    parametry pm;

    // PARAMETRY UKŁADU
    pm.n = 9; // układ ma n^2 węzłów 
    pm.dx = nm2au(2.0); // odległości międzywęzłowe

    // PARAMETRY KROPKI KWANTOWEJ
    pm.omega_x = eV2au(80e-3); // parametr skalujący potencjał w x
    pm.omega_y = eV2au(400e-3); // parametr skalujący potencjał w y

    // PARAMETRY ELEKTRONU
    pm.m = 0.24; // masa efekywna elektronu
    pm.E = 1.0; // energia elektronu
    pm.alpha_x = 1.0/pm.m/pm.omega_x; // wariancja gaussianu w x
    pm.alpha_y = 1.0/pm.m/pm.omega_y; // wariancja gaussianu w y
    
    // ----------------------------------------------------------
    
    /*plot_gaussian_v2(0, pm);
    plot_gaussian_v2(8, pm);
    plot_gaussian_v2(9, pm);*/

    
    pm.n = 9; // układ ma n^2 węzłów 
    pm.dx = nm2au(1.0);

    matrix S = S_matrix(pm);
    matrix K = K_matrix(pm);
    matrix V = V_matrix(pm);
    matrix E = K+V;

    std::pair<vector, matrix> energy_c = Hc_ESc(E, S, pm);
    for(int i = 0; i < energy_c.first.size(); i++){
        std::cout << "E[" << i << "] = " << energy_c.first[i]/eV2au(1.0) << "\n";
    }

    plot_psi_v2(energy_c.second, 0, pm);
    plot_psi_v2(energy_c.second, 1, pm);
    plot_psi_v2(energy_c.second, 2, pm);
    plot_psi_v2(energy_c.second, 3, pm);
    plot_psi_v2(energy_c.second, 4, pm);
    plot_psi_v2(energy_c.second, 5, pm);

    plot_10lowest(pm, 0.0, eV2au(500e-3), eV2au(5e-3));

}