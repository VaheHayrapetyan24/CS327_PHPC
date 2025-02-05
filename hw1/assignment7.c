#include <stdio.h>
#include <stdlib.h>

void print_string(char* str) {
	while(*str != '\0') 
		printf("%c", *(str++));
	printf("\n");
	return;
}

int main() {
	char* arr[10] = {};

	for (int i = 0; i < 10; ++i) {
		// I'm purposely allocating different lengths of memory
		arr[i] = (char*) malloc(sizeof(char) * i + 2);
		int j = 0;
		for (; j < i + 1; ++j) {
			arr[i][j] = '0' + j;
		}
		arr[i][j] = '\0';
		print_string(arr[i]);
	}

	// I want to use pointer arithmetic for motification part
	*(*(arr + 2) + 1) = '9'; // modifying 3rd string's 2nd element
	*(*(arr + 4) + 3) = '\n';
	*(*(arr + 9) + 5) = '\0';

	printf("modified \n");
	char** p = arr;
	while(p < arr + 10) {
		print_string(*p);
		free(*p); // freeing each string's memory
		p++;
	}
}
