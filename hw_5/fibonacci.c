#include <stdio.h>
#include <time.h>

#define NUM_THREADS 10

int fibbonacci_naive(int n) {
    if (n <= 2) {
        return 1;
    }


    return fibbonacci_naive(n - 1) + fibbonacci_naive(n - 2);
}

int fibbonaci_tasks(int n) {
    if (n <= 10) {
        return fibbonacci_naive(n);
    }

    int vals[2] = {0};

    for (int i = 1; i <= 2; ++i) {
        #pragma omp task shared(vals)
        {
            vals[i - 1] += fibbonaci_tasks(n - i);
        }
    }

    #pragma omp taskwait

    return vals[0] + vals[1];
}

int fibbonaci_parallel(int n) {
    int res;
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp single
        res = fibbonaci_tasks(n);
    }
    return res;
}



double get_time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}


int main() {
    int n = 45;

    double start = get_time_in_seconds();
    int fib = fibbonacci_naive(n);
    double end = get_time_in_seconds();
    printf("fibb naive %d %f \n", fib, (end - start));



    start = get_time_in_seconds();
    fib = fibbonaci_tasks(n);
    end = get_time_in_seconds();
    printf("fibb branched %d %f \n", fib, (end - start));
}