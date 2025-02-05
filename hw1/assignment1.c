#include <stdio.h>

int main() {
	int i = 5;
	int*p = &i;

	printf("%p %p \n", &i, p);
	(*p) += 9;

	printf("%d \n", *p);
}
