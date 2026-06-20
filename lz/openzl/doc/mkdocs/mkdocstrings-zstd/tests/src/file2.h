#ifndef FILE2_H
#define FILE2_H

#include "dir/file3.h"
#include "file1.h"

/**
 * A macro that forwards its argument.
 *
 * @param x an integer to forward
 * @returns @p x
 */
#define MACRO2(x) MACRO3(x)

typedef struct struct2_s MACRO1(struct2);

/**
 * A function that does something with @p s and @p x.
 *
 * @pre `s != NULL`
 * @pre `x > 0`
 * @post `s->x == x`
 * @param s a struct to operate on with type @ref struct2
 * @param x an integer to operate on
 * @return `s->x`
 */
int func2(struct2* s, int x);

typedef enum {
    /// The first enum value
    enum2_value1 = 0,
    /**
     * Another enum value that uses a macro @ref MACRO2
     */
    MACRO2(enum2_value2),
    MACRO1(enum2_value3),     //< third enum value
    MACRO3(enum2_value5) = 5, //< explicit initializer
} enum2;

#endif
