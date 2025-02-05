#include <stdio.h>

int main() {
	int a = 5;
	int *b = &a;

	int **c = &b;

	printf("%d %d\n", *b, **c);
}
