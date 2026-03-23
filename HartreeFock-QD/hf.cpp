#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cstdlib>
#include <random>
#include "hf.h"

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

vector operator+(const vector& v1, const vector& v2){

    size_t size1 = v1.size();
    size_t size2 = v2.size();
    if(size1 != size2){
        throw std::runtime_error("kurcze chlopaki cos nie tak z wektorami");
    }
    vector temp = vector(size1);
    for(int i = 0; i < size1; i++){
            temp[i] = v1[i]+v2[i];
    }
    return temp;
}

vector operator-(const vector& v1, const vector& v2){

    size_t size1 = v1.size();
    size_t size2 = v2.size();
    if(size1 != size2){
        throw std::runtime_error("kurka wodna panowie cos nie tak z macierzami");
    }
    vector temp = vector(size1);
    for(int i = 0; i < size1; i++){
            temp[i] = v1[i]-v2[i];
    }
    return temp;

}

vector operator*(double constant, const vector& vec){
    size_t size = vec.size();
    vector temp = vector(size);
    for(int i = 0; i < size; i++){
        temp[i] = constant * vec[i];
    }
    return temp;
}

// funkcje pomocnicze - fizyka

double eV2au(double eV){
    /* funkcja do konwersji eV -> a.u. */

    return 0.036749325871*eV;
}

double nm2au(double nm){
    /* funkcja do konwersji nm -> a.u. */

    return 18.897261339*nm;
}

double V(const vector& x, const parametry& pm){

    /* funkcja determinujące gęstość oddziaływania kulombowskiego;
    przyjmuje wektor std::vector<double> reprezentujący wartości x oraz y */

    return pm.kappa/std::sqrt( pow((x[0] - x[1]),2.0) + pm.l*pm.l );
}

// inne użyteczności

matrix random_matrix_NxN(int N){

    /* tworzy macierz rozmiaru (N,N) i wypełnia ją
    wartościami pseudolosowymi z zakresu -1 -> 1 
    poza brzegiem, gdzie wartości są równe 0 */

    matrix mat = matrix(N, vector(N, 0.0));
    for(int i = 1; i < N - 1; i++){
        for (int j = 1; j < N - 1; j++){
            mat[i][j] = random_uniform_m11(); // Inicjalizacja losowymi wartościami
        }
    }

    return mat;
}

vector random_vector_N(int N){

    /* tworzy wektor rozmiaru N i wypełnia go
    wartościami pseudolosowymi z zakresu -1 -> 1 
    poza krańcami, gdzie wartości są równe 0 */

    vector vec = vector(N, 0.0);
    for(int i = 1; i < N - 1; i++){
            vec[i] = random_uniform_m11(); // Inicjalizacja losowymi wartościami
    }

    return vec;
}

// MOJE - pewnie błędne
void antisymmetrize(vector& wave_function_1D, const parametry& pm){

    for(int i = 0; i < wave_function_1D.size(); i++){
        wave_function_1D[i] = 0.5*(wave_function_1D[i] - wave_function_1D[wave_function_1D.size() - 1 - i]);
    }

}

vector idx2xy(int i, int j, const parametry& pm){
    /* funkcja transformująca indeksy w macierzy
    na współrzędne położeniowe */

    return {-pm.a + pm.dx*i, -pm.a + pm.dx*j};
}

double idx2xy(int i, const parametry& pm){
    /* funkcja transformująca indeksy w wektorze
    na współrzędne położeniowe */

    return -pm.a + pm.dx*i;
}

inline double random_uniform_m11(){
    /** Mersenne Twister **/

    static std::random_device random;
    static std::mt19937 generator(random());
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    return -1.0 + 2.0*dist(generator);
}

void save_matrix(matrix& c, parametry& pm, std::ofstream& plik){

    if(!plik.is_open()){ return; }
    for(int i = 0; i < c.size(); i++){
        for(int j = 0; j < c[0].size(); j++){
            plik << c[i][j] << " ";
        }
        plik << "\n";
    }
    plik << "\n ";
    
    return;
}

void save_vector(vector& c, parametry& pm, std::ofstream& plik){

    if(!plik.is_open()){ return; }
    for(double element : c){
        plik << element << " ";
    }
    plik << "\n";

    //std::string command = "python3 lab1_psi.py";
    //int wynik = system(command.c_str());
    //remove("temp.dat");
    
    return;
}

// METODA CZASU UROJONEGO

matrix Hamiltonian(const matrix& wave_function, const parametry& pm){

    /* wave_function - macierz reprezentująca wartości funkcji falowej
    w węzłach jednowymiarowej studni potencjału, jest w postaci
    wave_function[x_1][x_2], tj. reprezentuje funkcję zmiennej
    położeniowej elektronu pierwszego x_1 oraz elektronu drugiego x_2;
    parametry to struktura definiujaca parametry ukladu;

    zwraca przetransformowaną operatorem Hamiltona funkcję falową */

    int size1 = wave_function.size();
    int size2 = wave_function[0].size();
    matrix H_wave_function = matrix(size1, vector(size2));

    for(int i = 0; i < size1; i++){ // WB Dirichleta: brzeg x_1
        H_wave_function[0][i] = 0.0;
        H_wave_function[size2-1][i] = 0.0;
    }
    for(int i = 0; i < size2; i++){ // WB Dirichleta: brzeg x_2
        H_wave_function[i][0] = 0.0;
        H_wave_function[i][size1-1] = 0.0;
    }
    for(int i = 1; i < size1-1; i++){
        for(int j = 1; j < size2-1; j++){ // Działanie Hamiltonianu

                H_wave_function[i][j] = -0.5*(wave_function[i+1][j]
                + wave_function[i-1][j] + wave_function[i][j+1] +
                wave_function[i][j-1] - 4*wave_function[i][j])/pm.dx/pm.dx/pm.m
                + V(idx2xy(i,j,pm), pm)*wave_function[i][j];
        }
    }

    return H_wave_function;
}

double calculate_energy(const matrix& wave_function, const parametry& pm){

    /* funkcja, która oblicza energię elektronu opisanego funkcją falową
    wave_function, jako <wave_function|Hamiltonian|wave_function> */

    double energy = 0.0;
    matrix H_wave_function = Hamiltonian(wave_function, pm);

    for(int i = 1; i < wave_function.size()-1; i++){
        for(int j = 1; j < wave_function[0].size()-1; j++){
            energy += wave_function[i][j]*H_wave_function[i][j]*pm.dx*pm.dx;
        }
    }
    return energy;
}

vector relax_in_imaginary_time(matrix& wave_function, const parametry& pm){

    /* funkcja służąca wykonaniu wielokronej relaksacji funkcji falowej wave_function
    z użyciem metody czasu urojonego z parametrami określonymi w pm,
    do osiągnięcia względnej różnicy energii kolejnych kroków na poziomie tol:
    |1- E_{n-1}/E_{n}| < tol;
    
    zwraca energie uzyskane w kolejnych iteracjach w postaci std::vector<double> */

    vector energies;
    double norm2;
    
    for(int iteration = 0; iteration < pm.N; iteration++){

        wave_function = wave_function - pm.dtau*Hamiltonian(wave_function, pm);

        for(int i = 0; i < wave_function.size(); i++){ // WB Dirichleta: brzeg x_1
            wave_function[0][i] = 0.0;
            wave_function[wave_function[0].size()-1][i] = 0.0;
        }
        for(int i = 0; i < wave_function[0].size(); i++){ // WB Dirichleta: brzeg x_2
            wave_function[i][0] = 0.0;
            wave_function[i][wave_function.size()-1] = 0.0;
        }

        norm2 = 0.0;
        for(int i = 1; i < wave_function.size()-1; i++){
            for(int j = 1; j < wave_function[0].size()-1; j++){  // Stała normalizacyjna
                norm2 += wave_function[i][j]*wave_function[i][j]*pm.dx*pm.dx;
            }
        }
        wave_function = (1.0/std::sqrt(norm2))*wave_function; // Normalizacja

        energies.push_back(calculate_energy(wave_function, pm));

        if(iteration > 0 && std::abs(1.0 - energies[iteration - 1]/energies[iteration]) < pm.tol){
            break;
        }
    }
    return energies;
}

// METODA HARTREE-FOCKA

vector HF_h(const vector& wave_function_1D, const parametry& pm){

    /* zwraca element macierzowy związany z energią kinetyczną
    na rzecz metody Hartee-Focka na podstawie wektora wave_function_1D */
    
    vector temp = vector(wave_function_1D.size(), 0.0);
    int size = wave_function_1D.size();

    for(int i = 0; i < size; i++){
            temp[i] = -0.5*(wave_function_1D[i-1]+wave_function_1D[i+1]-2.0*wave_function_1D[i])/pm.m/pm.dx/pm.dx;
    }

    return temp;

}

vector HF_J(const vector& wave_function_1D, const parametry& pm){

    /* zwraca element macierzowy "kulombowski" na rzecz
    metody Hartee-Focka na podstawie wektora wave_function_1D */
    
    vector temp = vector(wave_function_1D.size(), 0.0);
    vector delta_x;
    double x_i, x_j;

    for(int i = 1; i < wave_function_1D.size()-1; i++){
        x_i = idx2xy(i, pm);
        for(int j = 1; j < wave_function_1D.size()-1; j++){

            x_j = idx2xy(j, pm);
            delta_x = {x_i, x_j};

            temp[i] += wave_function_1D[j]*wave_function_1D[j]*V(delta_x, pm)*pm.dx;
        }

        temp[i] *= wave_function_1D[i];
    }

    return temp;

}

vector HF_F_operator(const vector& wave_function_1D, const parametry& pm){

    /* funkcja oblicza wektor będący wynikiem działania operatora
    Hartree-Focka na wektor wave_function_1D */

    vector temp = vector(wave_function_1D.size(), 0.0);
    temp = HF_h(wave_function_1D, pm) + HF_J(wave_function_1D, pm);

    return temp;

}

double HF_calculate_energy(const vector& wave_function_1D, const parametry& pm){

    /* funkcja, która oblicza energię elektronu opisanego funkcją falową
    wave_function_1D, jako <wave_function|Hamiltonian|wave_function> */

    double energy = 0.0;
    vector HF_F_wave_function_1D = HF_F_operator(wave_function_1D, pm);

    for(int i = 0; i < wave_function_1D.size(); i++){
            energy += pm.dx*wave_function_1D[i]*HF_F_wave_function_1D[i];
    }
    return energy;
}

vector HF_relax_in_imaginary_time(vector& wave_function_1D, const parametry& pm){

    /* funkcja służąca wykonaniu wielokronej relaksacji funkcji falowej wave_function_1D
    z użyciem metody czasu urojonego dla operatora Hartree-Focka z parametrami określonymi
    w pm do osiągnięcia względnej różnicy energii kolejnych kroków na poziomie tol:
    |1- E_{n-1}/E_{n}| < tol;
    
    zwraca energie uzyskane w kolejnych iteracjach w postaci std::vector<double> */

    vector energies, J;
    double E_J, norm2;

    for(int iteration = 0; iteration < pm.N; iteration++){

        //antisymmetrize(wave_function_1D, pm);
        wave_function_1D = wave_function_1D - pm.dtau*HF_F_operator(wave_function_1D, pm);

        wave_function_1D[wave_function_1D.size()-1] = 0.0; // WB Dirichleta: lewy koniec
        wave_function_1D[0] = 0.0; // WB Dirichleta: prawy koniec

        norm2 = 0.0;
        for(int i = 0; i < wave_function_1D.size(); i++){
            norm2 += wave_function_1D[i]*wave_function_1D[i]*pm.dx; // Stała normalizacyjna
        }
        wave_function_1D = (1.0/std::sqrt(norm2))*wave_function_1D; // Normalizacja
        
        J = HF_J(wave_function_1D, pm); // Element macierzowy "kulombowski"
        E_J = 0.0;
        for(int i = 0; i < wave_function_1D.size(); i++){
            E_J += wave_function_1D[i]*J[i]*pm.dx; // Energia <wave_function | J(wave_function)>
        }

        energies.push_back(2.0*HF_calculate_energy(wave_function_1D, pm) - E_J);

        if(iteration > 0 && std::abs(1.0 - energies[iteration - 1]/energies[iteration]) < pm.tol){
            break;
        }
    }

    return energies;
}

int main(){
    
    int N = 41;
    parametry pm;

    // PARAMETRY ELEKTRONU
    pm.m = 0.067; // masa efekywna elektronu
    pm.kappa = 1.0/12.5; // stała przy potencjale elektrycznym

    // PARAMETRY KROPKI KWANTOWEJ
    pm.a = nm2au(30.0); // długość QD
    pm.l = nm2au(10.0); //szerokość drutu, w którym znajduje się QD

    // PARAMETRY UKŁADU
    pm.dx = 2.0*pm.a/((double)N-1.0);

    // PARAMETRY SYMULACJI
    pm.N = 100000; // maksymalna liczba relaksacji
    pm.dtau = 0.067*pow(pm.dx, 2.0)*0.4; // krok w czasie urojonym relaksacji
    pm.tol = 1.0e-9;

    // ----------------------------------------------------------

    // ZADANIE 1.
    
    std::ofstream plik1("z1.dat");
    matrix zad1_psi = random_matrix_NxN(N);
    vector zad1_energies = relax_in_imaginary_time(zad1_psi, pm);

    save_vector(zad1_energies,  pm, plik1);
    plik1.close();

    // ZADANIE 2.

    std::ofstream plik2("z2.dat");
    vector zad2_energies_a, zad2_energies;
    matrix zad2_psi;
    for(pm.a = nm2au(30.0); pm.a <= nm2au(60.0); pm.a += nm2au(5.0)){

        pm.dx = 2.0*pm.a/((double)N-1.0);
        pm.dtau = 0.067*pow(pm.dx, 2.0)*0.4;

        zad2_psi = random_matrix_NxN(N);
        zad2_energies = relax_in_imaginary_time(zad2_psi, pm);
        zad2_energies_a.push_back(zad2_energies.back());
        zad2_psi.clear();
    }
    save_vector(zad2_energies_a, pm, plik2);
    plik2.close();
    
    // ZADANIE 4.
    
    std::ofstream plik4("z4.dat");
    vector zad4_energies_a, zad4_energies, zad4_psi;
    for(pm.a = nm2au(30.0); pm.a <= nm2au(60.0); pm.a += nm2au(5.0)){

        pm.dx = 2.0*pm.a/((double)N-1.0);
        pm.dtau = 0.067*pow(pm.dx, 2.0)*0.4;
        
        zad4_psi = random_vector_N(N);
        zad4_energies = HF_relax_in_imaginary_time(zad4_psi, pm);
        zad4_energies_a.push_back(zad4_energies.back());
        //zad4_psi.clear();
    }
    save_vector(zad4_energies_a,  pm, plik4);
    plik4.close();

    // funkcja falowa z użyciem H-F (nie było w treści projektu)
    std::ofstream plik_30("30.dat");
    save_vector(zad4_psi, pm, plik_30);
    plik_30.close();
    
    
    // ZADANIE 3.
    
    pm.a = nm2au(30.0); pm.dx = 2.0*pm.a/((double)N-1.0); pm.dtau = 0.067*pow(pm.dx, 2.0)*0.4;

    std::ofstream plik330("z3_30.dat");
    matrix zad3_psi30 = random_matrix_NxN(N);
    vector _1 = relax_in_imaginary_time(zad3_psi30, pm);

    save_matrix(zad3_psi30,  pm, plik330);
    plik330.close();

    pm.a = nm2au(60.0); pm.dx = 2.0*pm.a/((double)N-1.0); pm.dtau = 0.067*pow(pm.dx, 2.0)*0.4;

    std::ofstream plik360("z3_60.dat");
    matrix zad3_psi60 = random_matrix_NxN(N);
    vector _2 = relax_in_imaginary_time(zad3_psi60, pm);

    save_matrix(zad3_psi60,  pm, plik360);
    plik360.close();
    
}