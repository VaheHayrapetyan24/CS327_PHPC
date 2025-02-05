#include <stdio.h>

void print(int* a) {
	int* p = a;
	while (p < a + 5) {
                printf("%d \n", *(p++));
        }
}


int main() {
	int a[5] = {1,2,3,4,5};
	
	print(a);

	printf("------------\n");

	*(a) = 5;
	*(a + 1) = 4;
	*(a + 2) = 3;
	*(a + 3) = 2;
	*(a + 4) = 1;

	print(a);

	printf("-----------\n");

	for (int i = 0; i < 5; ++i) {
		printf("%d %d \n", a[i], *(a + i));
	}
}
