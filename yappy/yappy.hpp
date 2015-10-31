#include <stdint.h>

typedef uint8_t ui8;
typedef uint16_t ui16;
typedef uint32_t ui32;
typedef uint64_t ui64;

ui8 *YappyUnCompress(const ui8 *data, const ui8 *end, ui8 *to);

void YappyFillTables();

ui8 *YappyCompress(const ui8 *data, ui8 *to, size_t len, int level);