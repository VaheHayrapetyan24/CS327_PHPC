#include <stdio.h>
#include <stdlib.h>


int main() {
	int* a = malloc(sizeof(int));
	*a = 5;
	printf("%d\n", *a);

	int* b = malloc(sizeof(int[5])); // using two ways to allocate memory for 5-size array of int
	int* c =  malloc(sizeof(int) * 5);

	for (int i = 0; i < 5; ++i) {
		*(c + i) = i;
		b[i] = i; // just wanted to see if [] notation works on pointers

		printf("%d %d\n", b[i], *c);
	}

	free(a);
	free(b);
	free(c);

	// I wanted to see if free will also delete the values

	for (int i = 0; i < 5; ++i) {

                *c = i;
                *b = i;

                printf("%d %d\n", *(b++), *(c++));
        }
}
