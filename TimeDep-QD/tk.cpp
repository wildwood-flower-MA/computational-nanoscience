#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cstdlib>
#include <complex>
#include <random>
#include <map>
#include "eigen-3.4.0/Eigen/Dense"
#include "tk.h"

// zamiana jednostek na j.a.

double eV2au(double eV){
    /* funkcja do konwersji eV -> a.u. */

    return eV/27.211;
}

double nm2au(double nm){
    /* funkcja do konwersji nm -> a.u. */

    return nm/0.0529;
}

double kV_cm2au(double kV_cm){
    /* funkcja do konwersji kV/cm -> a.u. */

    constexpr double constant = 0.0001*0.0529/27.211;
    return constant*kV_cm;
}

double s_reversed2au(double s_reversed){
    /* funkcja do konwersji s -> a.u. */

    return s_reversed*(2.418884e-17);
}

// klasa System:
// sposób rozwiązania zadania przyjąłem taki:
// tworzę klasę zawierającą wszystkie informacje o układzie;
// posiada ona, poza położeniami punktów i potencjałem,
// przede wszystkim funkcję falową -
// ją z kolei można aktualizować używając dostępnych metod:
// metody macierzowej, metody C-N oraz A-C

System::System(const dict& parametry): parametry{parametry}, dx{2.0*parametry.at("a")/(parametry.at("N") + 1.0)}{

    int N = static_cast<int>(parametry.at("N"));
    this -> wavefunction = cvector(N, {0.0, 0.0});
    this -> positions = vector(N, 0.0);
    this -> potential = vector(N, 0.0);
    this -> F_potential = vector(N, 0.0);

    double a = parametry.at("a");
    double V1 = parametry.at("V1");
    double V2 = parametry.at("V2");
    double d1 = parametry.at("d1");
    double d2 = parametry.at("d2");
    double F = parametry.at("F");
    double position;

    // aktualizowanie wektorów położenia i potencjału el.
    for(int idx = 0; idx < N; idx++){
        position = -a + (static_cast<double>(idx)+1.0)*(this->dx);
        this->positions[idx] = position;
        this->F_potential[idx] = F*position;

        if(std::abs(position) >= std::abs(d1)){
            this->potential[idx] = V1; // jeśli |x|>|d1| => V = V1
        } else if(std::abs(position) <= std::abs(d2)){
            this->potential[idx] = V2;  // jeśli |x|<|d2| => V = V2
        } else { 
            this->potential[idx] = 0.0;  // w przeciwnym wypadku V = 0
        }
    }
}

Eigen::MatrixXcd eigenize_matrix(const cmatrix& mat){

    /* funkcja zamieniająca zwykłą macierz (wektor wektorów)
    na obiekt MatrixXcd biblioteki Eigen; potrzebne przy okazji
    diagonalizacji */

    int size = mat.size();
    Eigen::MatrixXcd matrix(size, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix(i, j) = mat[i][j];
        }
    }
    return matrix;
}

void System::print_potential(){

    /* metoda klasy System, wypisująca stałą część potencjału wzdłuż układu */

    for(int idx = 0; idx < (this->potential).size(); idx++){
        std::cout<< potential[idx] << " ";
    }
}

void System::reset_wavefunction(){

    /* zeruje funkcję falową */

    for(int idx = 0; idx < (this->wavefunction).size(); idx++){
        this->wavefunction[idx] = {0.0, 0.0};
    }
}

std::pair<vector, cmatrix> System::matrix_method(int n_modes, int mode_to_remember){

    /* metoda rozwiązująca problem elektronu w potencjale
    uwięzienia metodą macierzową; zwraca wektor posortowanych 
    energii elektronu i macierz wektorów własnych odpowiadających
    tym energiom */

    int N = static_cast<int>(this->parametry.at("N"));
    double dx = this->dx;
    double m_eff = this->parametry.at("m_eff");

    // tworzenie macierzy Hamiltonianu
    cmatrix hamiltonian = cmatrix(N, cvector(N, {0.0, 0.0}));
    std::complex<double> alpha = {0.5/m_eff/dx/dx, 0.0};
    std::complex<double> V;
    for(int idx = 0; idx < N; idx++){
        V = {(this->potential[idx]) + (this->F_potential[idx]), 0.0};
        hamiltonian[idx][idx] = 2.0*alpha + V;
        if(idx < N - 1){
            hamiltonian[idx][idx + 1] = -alpha;
            hamiltonian[idx + 1][idx] = -alpha;
        }
    }

    // znajdowanie wartości i wektorów własnych
    Eigen::MatrixXcd hamiltonian_eigen = eigenize_matrix(hamiltonian);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hamiltonian_eigen);
    Eigen::VectorXd energies_eigen = solver.eigenvalues();
    Eigen::MatrixXcd wavefunctions_eigen = solver.eigenvectors();

    // tworzenie par (wartość, wektor własny)
    std::vector<std::pair<double, Eigen::VectorXcd>> pairs_eigen;
    for (int idx = 0; idx < energies_eigen.size(); idx++){
        pairs_eigen.emplace_back(energies_eigen[idx], wavefunctions_eigen.col(idx));
    }

    // sortowanie par od najmniejszych energii
    auto to_sort = [](const auto& a, const auto& b){ return a.first < b.first; };
    std::sort(pairs_eigen.begin(), pairs_eigen.end(), to_sort);

    // zwraca tylko n_modes najniższych energetycznie
    vector to_return_energies = vector(n_modes, 0.0);
    cmatrix tr_wavefun = cmatrix(n_modes, cvector(pairs_eigen[0].second.size()));
    for(int idx = 0; idx < n_modes; idx++){
        to_return_energies[idx] = pairs_eigen[idx].first;
        for(int idxx = 0; idxx < pairs_eigen[idx].second.size(); idxx++){
            tr_wavefun[idx][idxx] = pairs_eigen[idx].second[idxx];
        }
    }

    // normowanie
    cvector norms = cvector(n_modes, {0.0, 0.0});
    for(int idx = 0; idx<n_modes; idx++){

        double norm = 0.0;
        for(int i = 0; i<tr_wavefun[idx].size(); i++){
            norm += std::norm(tr_wavefun[idx][i]);
        }
        norm = std::sqrt(norm*dx);
        for(int i = 0; i < tr_wavefun[idx].size(); i++){
            tr_wavefun[idx][i] /= norm;
        }
    }

    // klasa zapisuje stan mode_to_remember jako swoje wavefunction
    this->wavefunction = tr_wavefun[mode_to_remember];

    return {to_return_energies, tr_wavefun};

}

vector System::F_potential_in_time(double t){

    /* funkcja licząca potencjał w zależności od czasu */

    int N = static_cast<int>(this->parametry.at("N"));
    double omega = this->parametry.at("omega");
    vector F_potential_in_time = vector(N, 0.0);

    for(int idx = 0; idx < N; idx++){
        F_potential_in_time[idx] = (this->F_potential[idx])*std::sin(omega*t);
    }

    return F_potential_in_time;
}

std::pair<cvector, cvector> System::Crank_Nicholson(int n_iterations){

    /* metoda wykorzysująca schemat Cranka-Nicholson */

    int N = static_cast<int>(this->parametry.at("N"));
    double dx = this->dx;
    double m_eff = this->parametry.at("m_eff");

    // zamienia funkcję falową w układzie na funkcję falową
    // wynikającą z metody macierzowej
    std::pair<vector, cmatrix> _ = matrix_method(1, 0);

    std::complex<double> alpha = {0.5/m_eff/dx/dx, 0.0};

    // Hamiltonian(t = 0) oraz Hamiltonian(t = dt)
    cmatrix hamiltonian_t0 = cmatrix(N, cvector(N, {0.0, 0.0}));
    cmatrix hamiltonian_t1 = cmatrix(N, cvector(N, {0.0, 0.0})); 
    vector F_potential_t1 = this->F_potential_in_time(this->parametry.at("dt"));

    std::complex<double> V_t0, V_t1;

    // rzeczywiście, mógłbym całych tablic nie tworzyć - będzie to macierz rzadka,
    // niemniej tak jest łatwiej nie pogubić się później przy mnożeniu
    for(int idx = 0; idx < N; idx++){
        
        V_t0 = {(this->potential[idx]) + (this->F_potential[idx]), 0.0};
        hamiltonian_t0[idx][idx] = 2.0*alpha + V_t0;

        V_t1 = {(this->potential[idx]) + F_potential_t1[idx], 0.0};
        hamiltonian_t1[idx][idx] = 2.0*alpha + V_t1;

        if (idx < N - 1){
            hamiltonian_t0[idx][idx + 1] = -alpha;
            hamiltonian_t0[idx+1][idx] = -alpha;

            hamiltonian_t1[idx][idx + 1] = -alpha;
            hamiltonian_t1[idx+1][idx] = -alpha;
        }
    }

    int wf_size = this->wavefunction.size();
    cvector wavefunction_prime = cvector(wf_size, {0.0, 0.0});
    cvector wavefunction_local = cvector(wf_size, {0.0, 0.0});
    double dt = this->parametry.at("dt");

    cvector wavefunction_k = cvector(wf_size, {0.0, 0.0});
    for(int idx = 0; idx < wf_size; idx++){
        wavefunction_k[idx] = this->wavefunction[idx];
    }

    for(int iteration = 0; iteration < n_iterations; iteration++){

        for(int idx = 0; idx<wf_size; idx++){
            wavefunction_local[idx] = this->wavefunction[idx];
        }

        int next_idx, prev_idx;
        for(int idx = 1; idx < wf_size-1; idx++){

            // | a11 a12 0.0 | |v1|   |a11*v1 + a12*v2         |
            // | a21 a22 a23 |o|v2| = |a21*v1 + a22*v2 + a23*v3|
            // | 0.0 a32 a33 | |v3|   |a32*v2 + a33*v3         |
            wavefunction_prime[idx] = (hamiltonian_t0[idx][idx]*wavefunction_local[idx]
                                    + hamiltonian_t0[idx][idx + 1]*wavefunction_local[idx + 1]
                                    + hamiltonian_t0[idx][idx - 1]*wavefunction_local[idx - 1])
                                    + (hamiltonian_t1[idx][idx]*wavefunction_k[idx]
                                    + hamiltonian_t1[idx][idx + 1]*wavefunction_k[idx + 1]
                                    + hamiltonian_t1[idx][idx - 1]*wavefunction_k[idx - 1]);

        }

        wavefunction_prime[0] = (hamiltonian_t0[0][0]*wavefunction_local[0]
                                + hamiltonian_t0[0][1]*wavefunction_local[1])
                                + (hamiltonian_t1[0][0]*wavefunction_k[0]
                                + hamiltonian_t1[0][1]*wavefunction_k[1]);

        wavefunction_prime[N-1] = (hamiltonian_t0[N-1][N-1]*wavefunction_local[N-1]
                                + hamiltonian_t0[N-1][N-2]*wavefunction_local[N-2])
                                + (hamiltonian_t1[N-1][N-1]*wavefunction_k[N-1]
                                + hamiltonian_t1[N-1][N-2]*wavefunction_k[N-2]);
        
        // krok Cranka-Nicolson
        std::complex<double> constant = {0.0, -0.5*dt};
        for(int idx = 0; idx<wf_size; idx++){
            wavefunction_k[idx] = wavefunction_local[idx] + constant*wavefunction_prime[idx];
        }

    }

    // normalizacja
    double norm = 0.0;
    for(int i = 0; i < wavefunction_k.size(); i++){
        norm += std::norm(wavefunction_k[i]);
    }
    norm = std::sqrt(norm*dx);
    for(int i = 0; i < wavefunction_k.size(); i++){
        wavefunction_k[i] /= norm;
    }

    return {wavefunction_local, wavefunction_k};
}

cmatrix System::Askar_Cakmak(int time_steps, int save_every_time_steps){

    int N = static_cast<int>(this->parametry.at("N"));
    double dx = this->dx;
    double m_eff = this->parametry.at("m_eff");
    double dt = this->parametry.at("dt");
    std::complex<double> alpha = {0.5/m_eff/dx/dx, 0.0};

    // funkcja falowa w zerowym i pierwszym kroku czasowym
    std::pair<cvector, cvector> psi0_psi1 = Crank_Nicholson(10);
    std::complex<double> constant = {0.0, -2.0*dt};

    cmatrix psi_in_time = cmatrix(time_steps/save_every_time_steps, cvector(N, {0.0, 0.0}));
    cvector psi_t_minus_2 = psi0_psi1.first;
    cvector psi_t_minus_1 = psi0_psi1.second;
    cvector psi_t_current = cvector(N, {0.0, 0.0});

    cvector psi_prime;
    vector F_potential_time;
    int number = 0;

    for(int t_idx = 2; t_idx < time_steps; t_idx++){

        // obliczenie psi_prime na rzecz schematu Askara-Cakmaka
        psi_prime = cvector(N, {0.0, 0.0});
        F_potential_time = this->F_potential_in_time(static_cast<double>(t_idx)*dt);

        for(int idx = 1; idx < N - 1; idx++){
            psi_prime[idx] = -alpha*(psi_t_minus_1[idx + 1] + psi_t_minus_1[idx - 1] - 2.0*psi_t_minus_1[idx])
                            + (this->potential[idx] + F_potential_time[idx])*psi_t_minus_1[idx];
        }
        psi_prime[0] = -alpha*(psi_t_minus_1[1] - 2.0*psi_t_minus_1[0])
                            + (this->potential[0] + F_potential_time[0])*psi_t_minus_1[0];

        psi_prime[N - 1] = -alpha*(psi_t_minus_1[N - 2] - 2.0*psi_t_minus_1[N - 1])
                            + (this->potential[N - 1] + F_potential_time[N - 1])*psi_t_minus_1[N - 1];

        // zastosowanie schematu Askara-Cakmaka
        for (int idx = 0; idx < N; idx++) {
            psi_t_current[idx] = psi_t_minus_2[idx] + constant*psi_prime[idx];
        }

        psi_t_minus_2 = psi_t_minus_1;
        psi_t_minus_1 = psi_t_current;

        // normowanie
        double norm = 0.0;
        for(int i = 0; i<psi_t_minus_2.size(); i++){
            norm += std::norm(psi_t_minus_2[i]);
        }
        norm = std::sqrt(norm*dx);
        for(int i = 0; i<psi_t_minus_2.size(); i++){
            psi_t_minus_2[i] /= norm;
        }
        norm = 0.0;
        for(int i = 0; i<psi_t_minus_1.size(); i++){
            norm += std::norm(psi_t_minus_1[i]);
        }
        norm = std::sqrt(norm*dx);
        for(int i = 0; i<psi_t_minus_1.size(); i++){
            psi_t_minus_1[i] /= norm;
        }

        // zapis co save_every_time_steps
        if (t_idx%save_every_time_steps == 0){
            psi_in_time[number]=psi_t_minus_1;
            number++;
        }
    }

    // klasa zapisuje psi po time_steps krokach czasowych
    for(int idx = 0; idx < psi_t_minus_1.size(); idx++){
        this->wavefunction[idx] = psi_t_minus_1[idx];
    }

    return psi_in_time;
}

std::complex<double> dot_product(const cvector& psi1, const cvector& psi2, double dx){

    if(psi1.size()!=psi2.size()){
        return {999999.9, 999999.9};
    };
    std::complex<double> product = {0.0,0.0};
    for(int idx = 0; idx < psi1.size(); idx++){
        product += std::conj(psi2[idx])*psi1[idx];
    }
    return product*dx;
}

int main(){

    
    // tk 1.
    std::ofstream plik1("z1.dat");

    dict pm_tk1;
    pm_tk1["N"] = 99;
    pm_tk1["a"] = nm2au(25.0);
    pm_tk1["m_eff"] = 0.067;
    pm_tk1["d1"] = nm2au(12.0);
    pm_tk1["V1"] = eV2au(0.25);
    pm_tk1["d2"] = nm2au(4.0);
    pm_tk1["V2"] = eV2au(0.2);
    pm_tk1["omega"] = s_reversed2au(0.0);
    pm_tk1["dt"] = 1.0;

    std::pair<vector, cmatrix> E_psi1;
    for(double i = 0; i< 41.0; i+=1.0){
        pm_tk1["F"] = kV_cm2au(-2.0 + i*0.1);
        System sys1(pm_tk1);
        E_psi1 = sys1.matrix_method(4, 0);
        for(int j = 0; j<4; j++){
            plik1<<E_psi1.first[j]/eV2au(1.0)<<" ";
        }
        plik1<<"\n";
    }
    plik1.close();

    // tk 2.
    std::ofstream plik2("z2.dat");

    dict pm_tk2;
    pm_tk2["N"] = 99;
    pm_tk2["a"] = nm2au(25.0);
    pm_tk2["m_eff"] = 0.067;
    pm_tk2["d1"] = nm2au(12.0);
    pm_tk2["V1"] = eV2au(0.25);
    pm_tk2["d2"] = nm2au(4.0);
    pm_tk2["V2"] = eV2au(0.2);
    pm_tk2["omega"] = s_reversed2au(0.0);
    pm_tk2["dt"] = 1.0;
    pm_tk2["F"] = kV_cm2au(0.0);

    System sys2(pm_tk2);
    std::pair<vector, cmatrix> E_psi2;
    E_psi2 = sys2.matrix_method(4, 0);

    for(double element : sys2.potential){
        plik2 << element << " ";
    }
    plik2 << "\n";
    for(cvector psi_n : E_psi2.second){
        for(std::complex<double> element : psi_n){
            plik2 << element.real() << " ";
        }
        plik2 << "\n";
    }

    plik2.close();

    // tk 3.
    std::ofstream plik3("z3.dat");
    std::ofstream plik3_psi("z3_psi.dat");

    dict pm_tk3;
    pm_tk3["N"] = 99;
    pm_tk3["a"] = nm2au(25.0);
    pm_tk3["m_eff"] = 0.067;
    pm_tk3["d1"] = nm2au(12.0);
    pm_tk3["V1"] = eV2au(0.25);
    pm_tk3["d2"] = nm2au(4.0);
    pm_tk3["V2"] = eV2au(0.2);
    pm_tk3["omega"] = eV2au(0.1/1000); // eV2au(0.6843/1000); 
    pm_tk3["dt"] = 1.0;
    pm_tk3["F"] = kV_cm2au(0.08);

    System sys3(pm_tk3);

    std::pair<vector, cmatrix> E_psi3;
    E_psi3 = sys3.matrix_method(2, 0);

    cvector Psi_0 = E_psi3.second[0];
    cvector Psi_1 = E_psi3.second[1];
    cvector Psi_0_1;
    for(int idx = 0; idx < Psi_0.size(); idx++){
        Psi_0_1.push_back((Psi_0[idx] + Psi_1[idx])/std::sqrt(2));
    }

    cmatrix psi_in_time;
    psi_in_time = sys3.Askar_Cakmak(3000000, 10000);

    vector dot_products_0, dot_products_1, dot_products_0_1;
    double dx = sys3.dx;
    for(int idx = 0; idx < psi_in_time.size(); idx++){
        dot_products_0.push_back(std::norm(dot_product(Psi_0, psi_in_time[idx], dx)));
        dot_products_1.push_back(std::norm(dot_product(Psi_1, psi_in_time[idx], dx)));
        dot_products_0_1.push_back(std::norm(dot_product(Psi_0_1, psi_in_time[idx], dx)));
        for(int idxx = 0; idxx < psi_in_time[idx].size(); idxx++){
            plik3_psi << std::norm(psi_in_time[idx][idxx]) << " ";
        }
        plik3_psi << "\n";
    }
    for(int idx = 0; idx < dot_products_0.size(); idx++){
        plik3 << dot_products_0[idx] << " " << dot_products_1[idx] << " " << dot_products_0_1[idx] << "\n";
    }

    plik3.close();
    plik3_psi.close();
        
    std::ofstream plik4("z4.dat");

    dict pm_tk4;
    pm_tk4["N"] = 99;
    pm_tk4["a"] = nm2au(25.0);
    pm_tk4["m_eff"] = 0.067;
    pm_tk4["d1"] = nm2au(12.0);
    pm_tk4["V1"] = eV2au(0.25);
    pm_tk4["d2"] = nm2au(4.0);
    pm_tk4["V2"] = eV2au(0.2);
    pm_tk4["omega"] = eV2au(0.6843/1000); 
    pm_tk4["dt"] = 1.0;

    std::pair<vector, cmatrix> E_psi3;

    for(double iF = 0.0; iF <= 0.2; iF+=0.01){

        pm_tk4["F"] = kV_cm2au(iF);
        System sys4(pm_tk4);

        E_psi3 = sys4.matrix_method(2, 0);

        cvector Psi_0 = E_psi3.second[0];
        cvector Psi_1 = E_psi3.second[1];

        cmatrix psi_in_timee;
        psi_in_timee = sys4.Askar_Cakmak(3000000, 10000);

        double dx = sys4.dx;
        for(int idx = 0; idx < psi_in_timee.size(); idx++){
            plik4 << std::norm(dot_product(Psi_1, psi_in_timee[idx], dx)) << " ";
        }
        plik4 << "\n";
        std::cout << "Policzono dla: F = "<< iF;
    }

    plik4.close();
    */
}