#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BMPHeader;

typedef struct {
    uint32_t header_size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bpp;
    uint32_t compression;
    uint32_t image_size;
    int32_t x_ppm;
    int32_t y_ppm;
    uint32_t colors_used;
    uint32_t colors_important;
} BMPInfoHeader;
#pragma pack(pop)

// Function to read a BMP file into separate R, G, B arrays
void read_bmp(const char *filename, BMPHeader *header, BMPInfoHeader *info, uint8_t **r, uint8_t **g, uint8_t **b) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    // Read headers
    fread(header, sizeof(BMPHeader), 1, file);
    fread(info, sizeof(BMPInfoHeader), 1, file);

    if (header->type != 0x4D42 || info->bpp != 24 || info->compression != 0) {
        printf("Unsupported BMP format (must be 24-bit uncompressed)\n");
        fclose(file);
        return;
    }

    int width = info->width;
    int height = abs(info->height);
    int row_padded = (width * 3 + 3) & ~3;

    // Allocate separate arrays for R, G, B
    *r = (uint8_t *)malloc(width * height);
    *g = (uint8_t *)malloc(width * height);
    *b = (uint8_t *)malloc(width * height);
    if (!*r || !*g || !*b) {
        perror("Memory allocation failed");
        fclose(file);
        return;
    }

    fseek(file, header->offset, SEEK_SET);

    // Read pixel data row by row
    uint8_t *row = (uint8_t *)malloc(row_padded);
    for (int y = 0; y < height; y++) {
        fread(row, 1, row_padded, file);  // Read row with padding
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            (*b)[idx] = row[x * 3];       // Blue
            (*g)[idx] = row[x * 3 + 1];   // Green
            (*r)[idx] = row[x * 3 + 2];   // Red
        }
    }
    free(row);
    fclose(file);
}



// Write BMP using separate R, G, B arrays
void write_bmp(const char *filename, BMPHeader *header, BMPInfoHeader *info, uint8_t *r, uint8_t *g, uint8_t *b) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file for writing");
        return;
    }

    fwrite(header, sizeof(BMPHeader), 1, file);
    fwrite(info, sizeof(BMPInfoHeader), 1, file);

    int width = info->width;
    int height = abs(info->height);
    int row_padded = (width * 3 + 3) & ~3;

    uint8_t *row = (uint8_t *)malloc(row_padded);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            row[x * 3] = b[idx];     // Blue
            row[x * 3 + 1] = g[idx]; // Green
            row[x * 3 + 2] = r[idx]; // Red
        }
        fwrite(row, 1, row_padded, file);  // Write row with padding
    }

    free(row);
    fclose(file);
}


void naive(
	uint8_t *r,
	uint8_t *g,
	uint8_t *b,
	uint8_t *nr, 
	size_t size
) {
    for (int i = 0; i < size; ++i) {
        uint8_t gray = (uint8_t)(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]);
        nr[i] = gray;
    }
}

void simd(
	uint8_t *r,
	uint8_t *g,
	uint8_t *b,
	uint8_t *nr,
	size_t size
) {
	__m256i rs = _mm256_set1_epi32(153); // floor(0.299 * 512)
	__m256i gs = _mm256_set1_epi32(300); // floor(0.587 * 512)
	__m256i bs = _mm256_set1_epi32(58); // floor(0.114 * 512)

	int i = 0;
    for (; i <= size - 16; i += 16) {
        // 1. Load 16 uint8_t values into a 128-bit register
        __m128i u8_r = _mm_loadu_si128((__m128i*)&r[i]);
		// 2. Convert to 16-bit unsigned integers
        __m256i u16_r = _mm256_cvtepu8_epi16(u8_r);
		// 3. Convert lower and upper halves to 32-bit integers
        __m256i i32_r_low = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u16_r, 0));
        __m256i i32_r_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u16_r, 1));


		__m128i u8_g = _mm_loadu_si128((__m128i*)&g[i]);
		__m256i u16_g = _mm256_cvtepu8_epi16(u8_g);		
		__m256i i32_g_low = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u16_g, 0));
        __m256i i32_g_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u16_g, 1));

		__m128i u8_b = _mm_loadu_si128((__m128i*)&b[i]);
		__m256i u16_b = _mm256_cvtepu8_epi16(u8_b);
        __m256i i32_b_low = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u16_b, 0));
        __m256i i32_b_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u16_b, 1));


		__m256i res1 = _mm256_mul_epi32(rs, i32_r_low);
		res1 = _mm256_add_epi32(res1, _mm256_mul_epi32(gs, i32_g_low));
		res1 = _mm256_add_epi32(res1, _mm256_mul_epi32(bs, i32_b_low));
		// divide everything by 512: shift 9 bits to right
		res1 = _mm256_srli_epi32(res1, 9);



		__m256i res2 = _mm256_mul_epi32(rs, i32_r_high);
		res2 = _mm256_add_epi32(res2, _mm256_mul_epi32(gs, i32_g_high));
		res2 = _mm256_add_epi32(res2, _mm256_mul_epi32(bs, i32_b_high));
		res2 = _mm256_srli_epi32(res2, 9);


		res1 = _mm256_packus_epi32(res1, res2); // pack two 8 size 32 bit registers into 16 size 16 bits

        _mm256_storeu_si256((__m256i*)&nr[i], res1);
    }

    for (; i < size; ++i) {
        uint8_t gray = (uint8_t)(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]);
        nr[i] = gray;
    }
}


int main() {
    BMPHeader header;
    BMPInfoHeader info;
    uint8_t *r, *g, *b;

	uint8_t *nr; // naive results
	uint8_t *sr; // simd results

	


    // Read BMP file
    read_bmp("./colors.bmp", &header, &info, &r, &g, &b);
    if (!r || !g || !b) return 1;

	size_t size = info.width * abs(info.height);

	nr = (uint8_t *)malloc(size);
	sr = (uint8_t *)malloc(size);

	clock_t start = clock();
    naive(r, g, b, nr, size);
	clock_t end = clock();
	printf("Naive Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

	start = clock();
	simd(r, g, b, sr, size);
	end = clock();
	printf("SIMD Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);


    // Write grayscale BMP
    write_bmp("./naive_grayscale.bmp", &header, &info, nr, nr, nr);
	write_bmp("./simd_grayscale.bmp", &header, &info, sr, sr, sr);

    free(r);
    free(g);
    free(b);
    return 0;
}