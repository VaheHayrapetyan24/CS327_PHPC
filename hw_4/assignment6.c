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

	__m256i rs = _mm256_set1_epi16(76); // floor(0.299 * 256)
	__m256i gs = _mm256_set1_epi16(150); // floor(0.587 * 256)
	__m256i bs = _mm256_set1_epi16(29); // floor(0.114 * 256)

	__m256i zs = _mm256_setzero_si256();

	int i = 0;
    for (; i <= size - 16; i += 32) {
		__m256i u8_r = _mm256_loadu_si256((__m256i*)&r[i]);
		__m256i r_low = _mm256_unpacklo_epi8(u8_r, zs);
    	__m256i r_high = _mm256_unpackhi_epi8(u8_r, zs);


		__m256i u8_g = _mm256_loadu_si256((__m256i*)&g[i]);
		__m256i g_low = _mm256_unpacklo_epi8(u8_g, zs);
    	__m256i g_high = _mm256_unpackhi_epi8(u8_g, zs);

		__m256i u8_b = _mm256_loadu_si256((__m256i*)&b[i]);
		__m256i b_low = _mm256_unpacklo_epi8(u8_b, zs);
    	__m256i b_high = _mm256_unpackhi_epi8(u8_b, zs);


		__m256i res1 = _mm256_mullo_epi16(rs, r_low);
		res1 = _mm256_add_epi16(res1, _mm256_mullo_epi16(gs, g_low));
		res1 = _mm256_add_epi16(res1, _mm256_mullo_epi16(bs, b_low));
		// divide everything by 256: shift 8 bits to right
		res1 = _mm256_srli_epi16(res1, 8);



		__m256i res2 = _mm256_mullo_epi16(rs, r_high);
		res2 = _mm256_add_epi16(res2, _mm256_mullo_epi16(gs, g_high));
		res2 = _mm256_add_epi16(res2, _mm256_mullo_epi16(bs, b_high));
		res2 = _mm256_srli_epi16(res2, 8);

		res1 = _mm256_packus_epi16(res1, res2);

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
    read_bmp("./bigboy.bmp", &header, &info, &r, &g, &b);
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
    write_bmp("./naive_bigboy.bmp", &header, &info, nr, nr, nr);
	write_bmp("./simd_bigboy.bmp", &header, &info, sr, sr, sr);

    free(r);
    free(g);
    free(b);
	free(nr);
    free(sr);
	r = NULL;
	g = NULL;
	b = NULL;
	nr = NULL;
	sr = NULL;
	
    return 0;
}