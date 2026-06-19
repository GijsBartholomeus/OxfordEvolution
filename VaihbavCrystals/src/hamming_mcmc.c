#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint64_t s;
} rng_t;

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void rng_seed(rng_t *r, uint64_t seed) {
    if (seed == 0) seed = 0x123456789abcdefULL;
    r->s = seed;
    uint64_t x = seed;
    r->s = splitmix64(&x);
}

static uint64_t rng_u64(rng_t *r) {
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 2685821657736338717ULL;
}

static double rng_double_open(rng_t *r) {
    return ((rng_u64(r) >> 11) + 1.0) * (1.0 / 9007199254740993.0);
}

static uint32_t rng_bounded(rng_t *r, uint32_t n) {
    uint64_t x, lim = UINT64_MAX - (UINT64_MAX % n);
    do {
        x = rng_u64(r);
    } while (x >= lim);
    return (uint32_t)(x % n);
}

static int read_counts(const char *path, uint64_t **counts, uint32_t *q, uint64_t *total) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    uint64_t cap = 1024, n = 0, sum = 0;
    uint64_t *arr = (uint64_t *)malloc(cap * sizeof(uint64_t));
    if (!arr) {
        fclose(f);
        return -2;
    }
    while (1) {
        uint64_t v;
        int rc = fscanf(f, "%lu", &v);
        if (rc == EOF) break;
        if (rc != 1) {
            free(arr);
            fclose(f);
            return -3;
        }
        if (n == cap) {
            cap *= 2;
            uint64_t *tmp = (uint64_t *)realloc(arr, cap * sizeof(uint64_t));
            if (!tmp) {
                free(arr);
                fclose(f);
                return -2;
            }
            arr = tmp;
        }
        arr[n++] = v;
        sum += v;
    }
    fclose(f);
    if (n == 0 || n > UINT32_MAX) {
        free(arr);
        return -4;
    }
    *counts = arr;
    *q = (uint32_t)n;
    *total = sum;
    return 0;
}

static int write_snapshot(const char *path, const uint32_t *omega, uint32_t n, uint32_t q) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (q <= 256) {
        uint8_t *buf = (uint8_t *)malloc(n);
        if (!buf) {
            fclose(f);
            return -2;
        }
        for (uint32_t i = 0; i < n; i++) buf[i] = (uint8_t)omega[i];
        fwrite(buf, 1, n, f);
        free(buf);
    } else if (q <= 65536) {
        uint16_t *buf = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
        if (!buf) {
            fclose(f);
            return -2;
        }
        for (uint32_t i = 0; i < n; i++) buf[i] = (uint16_t)omega[i];
        fwrite(buf, sizeof(uint16_t), n, f);
        free(buf);
    } else {
        fwrite(omega, sizeof(uint32_t), n, f);
    }
    fclose(f);
    return 0;
}

static void init_ordered(uint32_t *omega, const uint64_t *counts, uint32_t q) {
    uint64_t pos = 0;
    for (uint32_t a = 0; a < q; a++) {
        for (uint64_t c = 0; c < counts[a]; c++) omega[pos++] = a;
    }
}

static void shuffle(uint32_t *omega, uint32_t n, rng_t *rng) {
    for (uint32_t i = n - 1; i > 0; i--) {
        uint32_t j = rng_bounded(rng, i + 1);
        uint32_t tmp = omega[i];
        omega[i] = omega[j];
        omega[j] = tmp;
    }
}

static int64_t full_S(const uint32_t *omega, uint32_t n, uint32_t d) {
    int64_t s = 0;
    for (uint32_t x = 0; x < n; x++) {
        uint32_t lx = omega[x];
        for (uint32_t bit = 0; bit < d; bit++) {
            uint32_t y = x ^ (1u << bit);
            if (y > x && omega[y] == lx) s++;
        }
    }
    return s;
}

static int delta_swap(const uint32_t *omega, uint32_t i, uint32_t j, uint32_t d) {
    uint32_t a = omega[i], b = omega[j];
    if (a == b) return 0;
    int delta = 0;
    for (uint32_t bit = 0; bit < d; bit++) {
        uint32_t y = i ^ (1u << bit);
        if (y == j) continue;
        uint32_t ly = omega[y];
        delta += (ly == b) - (ly == a);
    }
    for (uint32_t bit = 0; bit < d; bit++) {
        uint32_t y = j ^ (1u << bit);
        if (y == i) continue;
        uint32_t ly = omega[y];
        delta += (ly == a) - (ly == b);
    }
    return delta;
}

static int validate_delta(uint32_t *omega, uint32_t n, uint32_t d, rng_t *rng, int trials) {
    int64_t s0 = full_S(omega, n, d);
    for (int t = 0; t < trials; t++) {
        uint32_t i = rng_bounded(rng, n);
        uint32_t j = rng_bounded(rng, n - 1);
        if (j >= i) j++;
        int ds = delta_swap(omega, i, j, d);
        uint32_t tmp = omega[i];
        omega[i] = omega[j];
        omega[j] = tmp;
        int64_t s1 = full_S(omega, n, d);
        tmp = omega[i];
        omega[i] = omega[j];
        omega[j] = tmp;
        if (s1 - s0 != ds) {
            fprintf(stderr, "delta validation failed trial=%d i=%u j=%u delta=%d full=%ld\n",
                    t, i, j, ds, (long)(s1 - s0));
            return 1;
        }
    }
    return 0;
}

static int64_t run_sweep(uint32_t *omega, uint32_t n, uint32_t d, double temp, int64_t s, rng_t *rng,
                         uint64_t *accepted, uint64_t *nonnull, uint64_t *nulls) {
    for (uint32_t m = 0; m < n; m++) {
        uint32_t i = rng_bounded(rng, n);
        uint32_t j = rng_bounded(rng, n - 1);
        if (j >= i) j++;
        if (omega[i] == omega[j]) {
            (*nulls)++;
            continue;
        }
        (*nonnull)++;
        int ds = delta_swap(omega, i, j, d);
        int accept = 0;
        if (ds >= 0) {
            accept = 1;
        } else {
            double p = exp((double)ds / temp);
            if (rng_double_open(rng) < p) accept = 1;
        }
        if (accept) {
            uint32_t tmp = omega[i];
            omega[i] = omega[j];
            omega[j] = tmp;
            s += ds;
            (*accepted)++;
        }
    }
    return s;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s --d D --counts counts.tsv --temps temps.tsv --out dir --sweeps N "
            "--burn B --thin T --seed S --init random|ordered --validate-trials N\n",
            argv0);
}

int main(int argc, char **argv) {
    uint32_t d = 0;
    const char *counts_path = NULL, *temps_path = NULL, *out_dir = NULL, *init = "random";
    uint32_t sweeps = 100, burn = 20, thin = 1;
    uint64_t seed = 1;
    int validate_trials = 0;

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--d") == 0 && a + 1 < argc) d = (uint32_t)strtoul(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--counts") == 0 && a + 1 < argc) counts_path = argv[++a];
        else if (strcmp(argv[a], "--temps") == 0 && a + 1 < argc) temps_path = argv[++a];
        else if (strcmp(argv[a], "--out") == 0 && a + 1 < argc) out_dir = argv[++a];
        else if (strcmp(argv[a], "--sweeps") == 0 && a + 1 < argc) sweeps = (uint32_t)strtoul(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--burn") == 0 && a + 1 < argc) burn = (uint32_t)strtoul(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--thin") == 0 && a + 1 < argc) thin = (uint32_t)strtoul(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--seed") == 0 && a + 1 < argc) seed = strtoull(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "--init") == 0 && a + 1 < argc) init = argv[++a];
        else if (strcmp(argv[a], "--validate-trials") == 0 && a + 1 < argc) validate_trials = atoi(argv[++a]);
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!d || d > 31 || !counts_path || !temps_path || !out_dir || burn > sweeps || thin == 0) {
        usage(argv[0]);
        return 2;
    }

    uint64_t *counts = NULL, total64 = 0;
    uint32_t q = 0;
    if (read_counts(counts_path, &counts, &q, &total64) != 0) {
        fprintf(stderr, "failed to read counts: %s\n", counts_path);
        return 1;
    }
    if (total64 != (1ULL << d) || total64 > UINT32_MAX) {
        fprintf(stderr, "counts sum %lu does not match 2^d or exceeds uint32 indexing\n", (unsigned long)total64);
        free(counts);
        return 1;
    }
    uint32_t n = (uint32_t)total64;
    uint32_t *omega = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!omega) {
        fprintf(stderr, "failed to allocate omega for N=%u\n", n);
        free(counts);
        return 1;
    }

    rng_t rng;
    rng_seed(&rng, seed);
    init_ordered(omega, counts, q);
    if (strcmp(init, "random") == 0) shuffle(omega, n, &rng);
    else if (strcmp(init, "ordered") != 0) {
        fprintf(stderr, "unknown init: %s\n", init);
        free(omega);
        free(counts);
        return 2;
    }

    if (validate_trials > 0) {
        int bad = validate_delta(omega, n, d, &rng, validate_trials);
        if (bad) {
            free(omega);
            free(counts);
            return 1;
        }
    }

    char ts_path[4096], summary_path[4096], manifest_path[4096];
    snprintf(ts_path, sizeof(ts_path), "%s/timeseries.tsv", out_dir);
    snprintf(summary_path, sizeof(summary_path), "%s/summary.tsv", out_dir);
    snprintf(manifest_path, sizeof(manifest_path), "%s/snapshots.tsv", out_dir);
    FILE *temps = fopen(temps_path, "r");
    FILE *ts = fopen(ts_path, "w");
    FILE *summary = fopen(summary_path, "w");
    FILE *manifest = fopen(manifest_path, "w");
    if (!temps || !ts || !summary || !manifest) {
        fprintf(stderr, "failed to open IO files under %s\n", out_dir);
        return 1;
    }
    fprintf(ts, "temp_index\ttemp\tsweep\tS\tE\taccepted\tnonnull\tnull\n");
    fprintf(summary, "temp_index\ttemp\tmean_S\tvar_S\tmean_E\theat_capacity_S\taccept_rate\tnonnull_rate\truntime_seconds\n");
    fprintf(manifest, "temp_index\ttemp\tpath\tdtype\n");

    int64_t s = full_S(omega, n, d);
    const double norm = (double)n * (double)d;
    int temp_index = 0;
    double temp;
    int save_flag;
    double total_start = now_seconds();
    while (fscanf(temps, "%lf %d", &temp, &save_flag) == 2) {
        double start = now_seconds();
        uint64_t accepted = 0, nonnull = 0, nulls = 0, kept = 0;
        long double sum_s = 0.0L, sum_s2 = 0.0L, sum_e = 0.0L;
        for (uint32_t sw = 1; sw <= sweeps; sw++) {
            s = run_sweep(omega, n, d, temp, s, &rng, &accepted, &nonnull, &nulls);
            double e = -2.0 * (double)s / norm;
            fprintf(ts, "%d\t%.17g\t%u\t%ld\t%.17g\t%lu\t%lu\t%lu\n",
                    temp_index, temp, sw, (long)s, e,
                    (unsigned long)accepted, (unsigned long)nonnull, (unsigned long)nulls);
            if (sw > burn && ((sw - burn) % thin == 0)) {
                kept++;
                sum_s += (long double)s;
                sum_s2 += (long double)s * (long double)s;
                sum_e += (long double)e;
            }
        }
        long double mean_s = kept ? sum_s / kept : (long double)s;
        long double var_s = kept ? (sum_s2 / kept - mean_s * mean_s) : 0.0L;
        if (var_s < 0 && var_s > -1e-6L) var_s = 0;
        long double mean_e = kept ? sum_e / kept : (-2.0L * (long double)s / (long double)norm);
        double elapsed = now_seconds() - start;
        double acc_rate = nonnull ? (double)accepted / (double)nonnull : 0.0;
        double nonnull_rate = (double)nonnull / ((double)n * (double)sweeps);
        double heat = (double)(var_s / ((long double)n * (long double)temp * (long double)temp));
        fprintf(summary, "%d\t%.17g\t%.17Lg\t%.17Lg\t%.17Lg\t%.17g\t%.17g\t%.17g\t%.17g\n",
                temp_index, temp, mean_s, var_s, mean_e, heat, acc_rate, nonnull_rate, elapsed);
        fflush(summary);
        fflush(ts);

        if (save_flag) {
            char snap_path[4096];
            const char *dtype = (q <= 256) ? "uint8" : ((q <= 65536) ? "uint16" : "uint32");
            snprintf(snap_path, sizeof(snap_path), "%s/omega_T%03d.bin", out_dir, temp_index);
            if (write_snapshot(snap_path, omega, n, q) != 0) {
                fprintf(stderr, "failed to write snapshot %s\n", snap_path);
                return 1;
            }
            fprintf(manifest, "%d\t%.17g\t%s\t%s\n", temp_index, temp, snap_path, dtype);
            fflush(manifest);
        }
        temp_index++;
    }
    double total_elapsed = now_seconds() - total_start;
    fprintf(stderr, "completed %d temperatures in %.3f seconds, final S=%ld\n",
            temp_index, total_elapsed, (long)s);

    fclose(temps);
    fclose(ts);
    fclose(summary);
    fclose(manifest);
    free(omega);
    free(counts);
    return 0;
}
