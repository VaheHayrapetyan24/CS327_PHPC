#include <stdio.h>

void print(int* a, int size) {
	int* p = a;
	while (p < a + size) {
                printf("%d \n", *(p++));
        }
}


int main() {
	int a[5] = {1,2,3,4,5};
	
	print(a, 5);

	printf("------------\n");

	for (int i = 0; i < 5; ++i) 
		*(a + i) = 5 - i;

	print(a, 5);

	printf("-----------\n");

	for (int i = 0; i < 5; ++i) {
		printf("%d %d \n", a[i], *(a + i));
	}
}
