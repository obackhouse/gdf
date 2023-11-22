/*
 *  jk.c
 *  Functions to build J/K contributions to the Fock matrix.
 */

#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<stdio.h>
#include<complex.h>
#include<omp.h>
#include "cblas.h"


/*
 *  Function to perform a (1, 0, 2) transpose
 */
int64_t transpose_102(
    /* Input: */
    int64_t A,              /* Size of the first dimension */
    int64_t B,              /* Size of the second dimension */
    int64_t C,              /* Size of the third dimension */
    double complex *arr,    /* Input array (na, nb, nc) */
    /* Output: */
    double complex *out)    /* Output array (nb, na, nc) */
{
    int64_t ierr = 0;

    /* Allocate temporary variables */
    const int64_t AC = A * C;
    const int64_t BC = B * C;

    /* Perform transpose */
    for (size_t j = 0; j < B; j++) {
    for (size_t i = 0; i < A; i++) {
    for (size_t k = 0; k < C; k++) {
        out[j*AC + i*C + k] = arr[i*BC + j*C + k];
    }}}

    return ierr;
}


/*
 *  Compute K (exchange) matrix for k-point sampled integrals and
 *  density matrices:
 *
 *      K_{rs} = \sum_{L} \sum_{pq} (L|qs) (L|rp) D_{pq}
 *
 *  This function supports the computation of the density matrix for
 *  a subset of k-point pairs.  The density matrix should be passed
 *  at all k-points, and the output arrays will be filled for all
 *  k-point indices that are present as the first index in the pairs.
 */
int64_t cderi_get_k(
    /* Input: */
    int64_t nk,             /* Number of k-points */
    int64_t nkij,           /* Number of k-points pairs */
    int64_t nao,            /* Number of atomic orbitals */
    int64_t naux,           /* Number of auxiliary basis functions */
    int64_t *naux_slice,    /* Slice of auxiliary basis functions */
    double complex *cderi,  /* Three-center integrals (nkij, naux, nao, nao) */
    double complex *dm,     /* Density matrix (nk, nao, nao) */
    int64_t *kis,           /* Indices of first k-point in each pair */
    int64_t *kjs,           /* Indices of second k-point in each pair */
    /* Output: */
    double complex *vk)      /* K matrix (nk, nao, nao) */
{
    int64_t ierr = 0;

    /* Allocate temporary variables */
    const int64_t l0 = naux_slice[0];
    const int64_t l1 = naux_slice[1];
    const int64_t K = nk;
    const int64_t KP = nkij;
    const int64_t N = nao;
    const int64_t L = naux;
    const int64_t M = l1 - l0;
    const int64_t N2 = N * N;
    const int64_t K2 = K * K;
    const int64_t MN = M * N;
    const int64_t LN2 = L * N2;
    const int64_t MN2 = M * N2;
    const int64_t KN2 = K * N2;
    const double complex Z0 = 0.0;
    const double complex Z1 = 1.0;

    /* Find map between (ki, kj) and index ij - hacky but scales fine */
    int64_t *ij_map = calloc(K2, sizeof(int64_t));
    for (size_t ij = 0; ij < KP; ij++) {
        const size_t ki = kis[ij];
        const size_t kj = kjs[ij];
        ij_map[ki*K + kj] = ij;
    }

#pragma omp parallel
    {
        /* Allocate work arrays */
        double complex *work1 = calloc(MN2, sizeof(double complex));
        double complex *work2 = calloc(MN2, sizeof(double complex));
        double complex *vk_priv = calloc(KN2, sizeof(double complex));

        /* Build K matrix */
#pragma omp for reduction(+:ierr)
        for (size_t ij = 0; ij < KP; ij++) {
            const size_t ki = kis[ij];
            const size_t kj = kjs[ij];
            const size_t ji = ij_map[kj*K + ki];

            /* cderi(j, i, l, r, p) dm(i, p, q) -> work1(l, r, q) */
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MN, N, N, &Z1,
                        &(cderi[ji*LN2 + l0*N2]), N, &(dm[ki*N2]), N, &Z0, work1, N);

            /* work1(l, r, q) -> work2(r, l, q) */
            transpose_102(M, N, N, &(work1[0]), &(work2[0]));

            /* work3(r, l, q) cderi(i, j, l, q, s) -> vk_priv(j, r, s) */
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, MN, &Z1,
                        work2, MN, &(cderi[ij*LN2 + l0*N2]), N, &Z1, &(vk_priv[kj*N2]), N);
        }

        /* Deallocate work arrays */
        free(work1);
        free(work2);

        /* Accumulate J and K matrices */
#pragma omp critical
        {
            for (size_t i = 0; i < KN2; i++) {
                vk[i] += vk_priv[i] / K;
            }
        }

        /* Deallocate private work arrays */
        free(vk_priv);
    }

    return ierr;
}
