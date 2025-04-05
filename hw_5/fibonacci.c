#include <stdio.h>
#include <time.h>

#define NUM_THREADS 10

int fibbonacci_naive(int n) {
    if (n <= 2) {
        return 1;
    }


    return fibbonacci_naive(n - 1) + fibbonacci_naive(n - 2);
}



int fibbonaci_wait_tasks(int n) {
    if (n <= 10) {
        return fibbonacci_naive(n);
    }

    int res;
    #pragma omp taskwait // this does nothing
    {
        res = fibbonacci_naive(n - 1) + fibbonacci_naive(n - 2);
    }

    return res;
}

int fibbonaci_sequential_parallel(int n) {
    int res;
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        res = fibbonaci_wait_tasks(n);
    }
    return res;
}


// this is good version
// int fibbonaci_branched_tasks(int n) {
//     if (n <= 10) {
//         return fibbonacci_naive(n);
//     }

//     int sum = 0;

//     #pragma omp task shared(sum)
//     {
//         // printf("sum task 1 before %d\n", sum);
//         #pragma omp atomic
//         sum += fibbonaci_branched_tasks(n - 1);
//         // printf("sum task 1 after %d\n", sum);
//     }

//     #pragma omp task shared(sum)
//     {
//         // printf("sum task 2 before %d\n", sum);
//         #pragma omp atomic
//         sum += fibbonaci_branched_tasks(n - 2);
//         // printf("sum task 2 after %d\n", sum);
//     }

    

//     #pragma omp taskwait

//     // printf("sum %d\n", sum);


//     return sum;
// }

// int fibbonaci_branched(int n) {
//     int res;
//     #pragma omp parallel num_threads(NUM_THREADS)
//     {
//         #pragma omp single
//         res = fibbonaci_branched_tasks(n);

//     }
//     return res;
// }


// this is what the idiot did
// int fibbonaci_branched_tasks(int n) {
//     if (n <= 10) {
//         return fibbonacci_naive(n);
//     }

//     int x = 0, y = 0;

//     #pragma omp task shared(x)
//     x = fibbonaci_branched_tasks(n - 1);

//     #pragma omp task shared(y)
//     y = fibbonaci_branched_tasks(n - 2);

//     #pragma omp taskwait
//     return x + y;
// }

// int fibbonaci_branched(int n) {
//     int res = 0;

//     #pragma omp parallel
//     {
//         #pragma omp single
//         res = fibbonaci_branched_tasks(n);
//     }

//     return res;
// }

int fibbonaci_branched_tasks(int n) {
    if (n <= 10) {
        return fibbonacci_naive(n);
    }

    int vals[2] = {0};

    for (int i = 1; i <= 2; ++i) {
        #pragma omp task shared(vals)
        {
            vals[i - 1] += fibbonaci_branched_tasks(n - i);
        }
    }

    #pragma omp taskwait

    return vals[0] + vals[1];
}

int fibbonaci_branched(int n) {
    int res;
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp single
        res = fibbonaci_branched_tasks(n);
    }
    return res;
}



double get_time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}


int main() {
    // for (int i = 5; i < 6; ++i) 

    int n = 50;
    double start = get_time_in_seconds();
    // int fib = fibbonaci_sequential_parallel(n);
    int fib;
	
	double end = get_time_in_seconds();
    printf("fibb parallel %d %f \n", fib, (end - start));
    
    start = get_time_in_seconds();
    // fib = fibbonacci_naive(n);
    end = get_time_in_seconds();
    printf("fibb naive %d %f \n", fib, (end - start));



    start = get_time_in_seconds();
    fib = fibbonaci_branched_tasks(n);
    end = get_time_in_seconds();
    printf("fibb branched %d %f \n", fib, (end - start));
}