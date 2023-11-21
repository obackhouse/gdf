/*
 *  jk.c
 *  Functions to build J/K contributions to the Fock matrix.
 */

#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
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
 *  Compute J (Coulomb) and K (exchange) matrices for k-point sampled
 *  integrals and density matrices:
 *
 *      J_{rs} = \sum_{L} \sum_{pq} (L|pq) (L|rs) D_{pq}
 *      K_{rs} = \sum_{L} \sum_{pq} (L|qs) (L|rp) D_{pq}
 */
int64_t cderi_jk(
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
    bool with_j,            /* Compute J matrix */
    bool with_k,            /* Compute K matrix */
    /* Output: */
    double complex *vj,      /* J matrix (nk, nao, nao) */
    double complex *vk)      /* K matrix (nk, nao, nao) */
{
    int64_t ierr = 0;
    if (!with_j && !with_k) return ierr;

    /* Allocate temporary variables */
    const int64_t l0 = naux_slice[0];
    const int64_t l1 = naux_slice[1];
    const int64_t K = nk;
    const int64_t N = nao;
    const int64_t L = naux;
    const int64_t M = l1 - l0;
    const int64_t N2 = N * N;
    const int64_t K2 = K * K;
    const int64_t MN = M * N;
    const int64_t LN2 = L * N2;
    const int64_t MN2 = M * N2;
    const int64_t KN2 = K * N2;
    const int64_t KLN2 = K * L * N2;
    const int64_t I1 = 1;
    const double complex Z0 = 0.0;
    const double complex Z1 = 1.0;

    /* Conjugate density matrix to avoid platform dependent CblasConjNoTrans */
    double complex *dm_conj = calloc(KN2, sizeof(double complex));
    for (size_t i = 0; i < KN2; i++) {
        dm_conj[i] = conj(dm[i]);
    }

#pragma omp parallel
    {
        double complex *vj_priv, *vk_priv;

        if (with_j) {
            /* Allocate work arrays */
            double complex *work1 = calloc(M, sizeof(double complex));
            vj_priv = calloc(KN2, sizeof(double complex));

            /* Build J matrix */
#pragma omp for reduction(+:ierr)
            for (size_t i = 0; i < K; i++) {
                // cderi(i, i, l, p, q) dm(i, p, q)* -> work1(l)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, I1, N2, &Z1,
                            &(cderi[i*KLN2 + i*LN2 + l0*N2]), N2, &(dm_conj[i*N2]), I1, &Z0, work1, I1);
            }

            for (size_t j = 0; j < K; j++) {
                // work1(l) cderi(j, j, l, r, s) -> vj_priv(j, r, s)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I1, N2, M, &Z1,
                            work1, M, &(cderi[j*KLN2 + j*LN2 + l0*N2]), N2, &Z1, &(vj_priv[j*N2]), N2);
            }

            /* Deallocate work arrays */
            free(work1);
        }

        if (with_k) {
            /* Allocate work arrays */
            double complex *work2 = calloc(MN2, sizeof(double complex));
            double complex *work3 = calloc(MN2, sizeof(double complex));
            vk_priv = calloc(KN2, sizeof(double complex));

            /* Build K matrix */
#pragma omp for reduction(+:ierr)
            for (size_t ij = 0; ij < K2; ij++) {
                size_t i = ij / K;
                size_t j = ij % K;

                // cderi(j, i, l, r, p) dm(i, p, q) -> work2(l, r, q)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MN, N, N, &Z1,
                            &(cderi[j*KLN2 + i*LN2 + l0*N2]), N, &(dm[i*N2]), N, &Z0, work2, N);

                // work2(l, r, q) -> work3(r, l, q)
                transpose_102(M, N, N, &(work2[0]), &(work3[0]));

                // work3(r, l, q) cderi(i, j, l, q, s) -> vk_priv(j, r, s)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, MN, &Z1,
                            work3, MN, &(cderi[i*KLN2 + j*LN2 + l0*N2]), N, &Z1, &(vk_priv[j*N2]), N);
            }

            /* Deallocate work arrays */
            free(work2);
            free(work3);
        }

        /* Accumulate J and K matrices */
#pragma omp critical
        {
            for (size_t i = 0; i < KN2; i++) {
                if (with_j) vj[i] += vj_priv[i] / K;
                if (with_k) vk[i] += vk_priv[i] / K;
            }
        }

        /* Deallocate private work arrays */
        if (with_j) free(vj_priv);
        if (with_k) free(vk_priv);
    }

    /* Deallocate temporary variables */
    free(dm_conj);

    return ierr;
}
