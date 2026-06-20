#ifndef DIR_FILE3_H
#define DIR_FILE3_H

#include "file1.h"
#include "file4.h"

/**
 * A macro that forwards its argument.
 *
 * @param x an integer to forward
 * @returns @p x
 */
#define MACRO3(x) MACRO1(x)

typedef struct struct3_s MACRO1(struct3);

/**
 * A function that does something with @p s and @p x.
 *
 * @pre `s != NULL`
 * @pre `x > 0`
 * @post `s->x == x`
 * @param s a struct to operate on with type @ref struct3
 * @param x an integer to operate on
 * @return `s->x`
 */
int func3(struct3* s, int x);

/// A const variable equal to 5
static const int MACRO4(var, 3) = 5;

typedef enum {
    /// The first enum value
    enum3_value1 = 0,
    /**
     * Another enum value that uses a macro @ref MACRO3
     */
    MACRO3(enum3_value2),
    MACRO1(enum3_value3),     //< third enum value with @ref MACRO1
    MACRO3(enum3_value5) = 5, //< explicit initializer
} enum3;

#endif
