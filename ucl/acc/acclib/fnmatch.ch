/* ACC -- Automatic Compiler Configuration

   Copyright (C) 1996-2004 Markus Franz Xaver Johannes Oberhumer
   All Rights Reserved.

   This software is a copyrighted work licensed under the terms of
   the GNU General Public License. Please consult the file "ACC_LICENSE"
   for details.

   Markus F.X.J. Oberhumer
   <markus@oberhumer.com>
   http://www.oberhumer.com/
 */


#define __ACCLIB_FNMATCH_CH_INCLUDED 1
#if !defined(ACCLIB_PUBLIC)
#  define ACCLIB_PUBLIC(r,f)    r __ACCLIB_FUNCNAME(f)
#endif


/*************************************************************************
//
**************************************************************************/

typedef struct {
    const acc_hbyte_p s;
    const acc_hbyte_p s_start;
    int f;
    int f_escape;
    int f_casefold;
} acc_fnmatch_internal_t;


static int __ACCLIB_FUNCNAME(acc_fnmatch_internal) (const acc_hbyte_p p, acc_fnmatch_internal_t* a)
{
    const acc_hbyte_p s = a->s;
    int f_pathname = (a->f & (ACC_FNMATCH_PATHNAME | ACC_FNMATCH_PATHSTAR));

    while (*p) {
        switch (*p) {
        case '?': case '[':
            if (*s == 0) return 0;
            if (*s == '/' && f_pathname) return 0;
            if (*s == '.' && (a->f & ACC_FNMATCH_PERIOD) && (s == a->s_start || (f_pathname && s[-1] == '/'))) return 0;
            if (*p == '?') break;
            {
            int r = 0, fail = 0; unsigned char last = 0;
            if (*++p == '^' || *p == '!')
                fail = 1, ++p;
            do {
                switch (*p) {
                case 0:
                    return -1;
                case '-':
                    if (last == 0 || p[1] == ']') goto acc_label_default;
                    if (a->f_escape && p[1] == '\\') ++p;
                    if (*++p == 0) return -1;
                    if (!a->f_casefold && last <= *s && *s <= *p) r = 1;
                    else if (a->f_casefold && acc_ascii_tolower(last) <= acc_ascii_tolower(*s) && acc_ascii_tolower(*s) <= acc_ascii_tolower(*p)) r = 1;
                    last = 0;
                    continue;
                /* TODO: implement sets like [:alpha:] ??? */
                case '\\':
                    if (a->f_escape && *++p == 0) return -1;
                default: acc_label_default:
                    if (*s == *p) r = 1;
                    else if (a->f_casefold && acc_ascii_tolower(*s) == acc_ascii_tolower(*p)) r = 1;
                    break;
                }
                last = *p;
            } while (*++p != ']');
            if (r == fail)
                return 0;
            break;
        }
        case '*':
            while (*++p == '*') if (a->f & ACC_FNMATCH_PATHSTAR) f_pathname = 0;
            if (*s == '.' && (a->f & ACC_FNMATCH_PERIOD) && (s == a->s_start || (f_pathname && s[-1] == '/'))) return 0;
            if (*p == 0) {
                if (f_pathname) while (*s) if (*s++ == '/') return 0;
                return 1;
            }
            for ( ; *s; ++s) {
                int r;
                a->s = s; r = __ACCLIB_FUNCNAME(acc_fnmatch_internal)(p, a);
                switch (r) {
                case 0:  if (*s == '/' && f_pathname) return 2; break;
                case 2:  if (!f_pathname) break;
                default: return r;
                }
            }
            return 0;
        case '\\':
            if (a->f_escape && *++p == 0) return -1;
        default:
            if (*s == *p) break;
            if (a->f_casefold && acc_ascii_tolower(*s) == acc_ascii_tolower(*p)) break;
            return 0;
        }
        ++p, ++s;
    }
    return *s == 0;
}


ACCLIB_PUBLIC(int, acc_fnmatch) (const acc_hchar_p p, const acc_hchar_p s, int flags)
{
    int r;
    acc_fnmatch_internal_t args;
    args.s = args.s_start = (const acc_hbyte_p) s;
    args.f = flags;
    args.f_escape = !(flags & ACC_FNMATCH_NOESCAPE);
    args.f_casefold = (flags & ACC_FNMATCH_ASCII_CASEFOLD);
    r = __ACCLIB_FUNCNAME(acc_fnmatch_internal)((const acc_hbyte_p)p, &args);
    if (r < 0) return r;
    return r != 1;
}


/*
vi:ts=4:et
*/
