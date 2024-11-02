# proyecto-algoritmos-paralelos---solucion-paralela-en-CUDA-del-rompecabezas-de-futoshiki

%%writefile proyecto2.cu
#include "device_launch_parameters.h"
#include <iostream>

#define N 5  // Tamaño del tablero 5x5

__constant__ char dev_restricciones_fila[N][N];
__constant__ char dev_restricciones_columna[N][N];

// Verificar si un número es válido en la posición dada
__device__ bool es_valido(int tablero[N][N], int fila, int columna, int num) {
    for (int i = 0; i < N; i++) {
        if (tablero[fila][i] == num || tablero[i][columna] == num)
            return false;
    }

    if (columna > 0 && dev_restricciones_fila[fila][columna - 1] == '>' && tablero[fila][columna - 1] <= num)
        return false;
    if (columna > 0 && dev_restricciones_fila[fila][columna - 1] == '<' && tablero[fila][columna - 1] >= num)
        return false;

    if (fila > 0 && dev_restricciones_columna[fila - 1][columna] == '>' && tablero[fila - 1][columna] <= num)
        return false;
    if (fila > 0 && dev_restricciones_columna[fila - 1][columna] == '<' && tablero[fila - 1][columna] >= num)
        return false;

    return true;
}

// Kernel para resolver el tablero Futoshiki
__global__ void resolver_futoshiki(int* solucion_encontrada, int tablero[N][N]) {
    int fila = 0;
    int columna = 0;

    while (fila < N && !(*solucion_encontrada)) {
        bool colocado = false;

        for (int num = 1; num <= N; num++) {
            if (es_valido(tablero, fila, columna, num)) {
                tablero[fila][columna] = num;
                colocado = true;
                break;
            }
        }

        if (!colocado) {
            tablero[fila][columna] = 0;

            if (columna == 0) {
                columna = N - 1;
                fila--;
            } else {
                columna--;
            }
            if (fila < 0) break;
        } else {
            if (fila == N - 1 && columna == N - 1) {
                atomicExch(solucion_encontrada, 1);
                return;
            } else {
                columna++;
                if (columna == N) {
                    columna = 0;
                    fila++;
                }
            }
        }
    }
}

int main() {
    int h_tablero[N][N] = {
        {0, 3, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    char h_restricciones_fila[N][N] = {
        {' ', '>', ' ', ' ', ' '},
        {' ', ' ', '<', ' ', ' '},
        {' ', ' ', ' ', '>', ' '},
        {' ', ' ', ' ', ' ', ' '},
        {' ', '<', ' ', ' ', ' '}
    };

    char h_restricciones_columna[N][N] = {
        {' ', ' ', ' ', ' ', ' '},
        {' ', ' ', '>', ' ', ' '},
        {' ', '<', ' ', ' ', ' '},
        {' ', ' ', ' ', ' ', ' '},
        {' ', ' ', ' ', ' ', ' '}
    };

    cudaMemcpyToSymbol(dev_restricciones_fila, h_restricciones_fila, N * N * sizeof(char));
    cudaMemcpyToSymbol(dev_restricciones_columna, h_restricciones_columna, N * N * sizeof(char));

    int* d_solucion_encontrada;
    int (*d_tablero)[N];
    int h_solucion_encontrada = 0;

    cudaMalloc(&d_solucion_encontrada, sizeof(int));
    cudaMalloc(&d_tablero, N * N * sizeof(int));
    cudaMemcpy(d_tablero, h_tablero, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_solucion_encontrada, &h_solucion_encontrada, sizeof(int), cudaMemcpyHostToDevice);

    resolver_futoshiki<<<1, 1>>>(d_solucion_encontrada, d_tablero);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tablero, d_tablero, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_solucion_encontrada, d_solucion_encontrada, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_solucion_encontrada) {
        std::cout << "Tablero resuelto:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << h_tablero[i][j] << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "No se encontró solución para el tablero dado.\n";
    }

    cudaFree(d_solucion_encontrada);
    cudaFree(d_tablero);
    return 0;
}




