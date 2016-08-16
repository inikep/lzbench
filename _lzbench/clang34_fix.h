// fix for clang version 3.4 from https://bugzilla.redhat.com/show_bug.cgi?id=1133508
#if !defined __extern_always_inline && defined __clang__
# if defined __GNUC_STDC_INLINE__ || defined __GNUC_GNU_INLINE__
#  define __extern_inline extern __inline __attribute__ ((__gnu_inline__))
#  define __extern_always_inline \
  extern __always_inline __attribute__ ((__gnu_inline__))
# else
#  define __extern_inline extern __inline
#  define __extern_always_inline extern __always_inline
# endif
#endif
