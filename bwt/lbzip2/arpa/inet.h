/*
  Shim for the three lbzip2 sources that include <arpa/inet.h>, which MinGW
  does not have.  They use nothing from it but ntohl() and htonl(), so this
  provides those and lets the vendored sources stay byte-for-byte upstream.
  Only reachable through -Ibwt/lbzip2, i.e. only for those sources.
*/
#ifndef LBZIP2_LZBENCH_ARPA_INET_H
#define LBZIP2_LZBENCH_ARPA_INET_H

#include <stdint.h>

/* Some libcs define these from headers lbzip2 already includes; leave those
   alone. */
#ifndef ntohl
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define ntohl(x) ((uint32_t)(x))
#else
#define ntohl(x) __builtin_bswap32((uint32_t)(x))
#endif
#endif

#ifndef htonl
#define htonl(x) ntohl(x)
#endif

#endif
