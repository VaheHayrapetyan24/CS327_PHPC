#include <stdio.h>


unsigned int str_length(char* str) {
	unsigned int i = 0;

	// just going right till I find '\0'
	while(*(str + (i++)) != '\0'){}

	// returning -1 to reflect the actual size of the string
	return i - 1;
}


int main () {
	char* c = "abcdef";
	char* p = c;

	while(*p != '\0') {
		printf("%c", *(p++));
	}

	printf("\n");


	printf("%u \n",str_length(c));

	char* user_input;

	// I want to use double quotes before and after the user's input
	// but after you enter after scanf it goes to a new line
	printf("please input a string: \"");
	scanf("%s", user_input);

	// so my plan is to move the cursor up, then calculate how many characters I need to go to the right
	printf("\033[A");

	// I create this string that will be something like \033[45C and this will move the cursor 45 chars to the right
	// so I calculate how long the string on the line is, "please input a string: \"" is 24 chars long, and I add the length of the input

	char *go_to_end;
	sprintf(go_to_end, "\033[%dC", 24 + str_length(user_input));

	// here I just pass 1 as an argument because there's a warning if you don't pass any args to printf
	printf(go_to_end, 1);

	printf("\"\nyour input length is: %u\n", str_length(user_input));


}
