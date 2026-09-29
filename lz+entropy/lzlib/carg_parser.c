/* Arg_parser - POSIX/GNU command-line argument parser. (C version)
   Copyright (C) 2006-2026 Antonio Diaz Diaz.

   This library is free software. Redistribution and use in source and
   binary forms, with or without modification, are permitted provided
   that the following conditions are met:

   1. Redistributions of source code must retain the above copyright
   notice, this list of conditions, and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "carg_parser.h"


static char is_number( const char * const p )
  {
  return isdigit( (unsigned char)(p[*p=='.']) ) ||
         strcmp( p, "inf" ) == 0 || strcmp( p, "Inf" ) == 0 ||
         strcmp( p, "INF" ) == 0;
  }


/* assure at least a minimum size for buffer 'buf' */
static void * ap_resize_buffer( void * buf, const int min_size )
  {
  if( buf ) buf = realloc( buf, min_size );
  else buf = malloc( min_size );
  return buf;
  }


static char set_argument( Arg_parser * const ap, const char * const argument )
  {
  ap_Record * const p = &(ap->data[ap->data_size-1]);
  const int len = strlen( argument );
  p->argument = (char *)malloc( len + 1 );
  if( !p->argument ) return 0;
  memcpy( p->argument, argument, len + 1 );
  return 1;
  }


static ap_Record * push_back_record( Arg_parser * const ap )
  {
  const int min_size = ( ap->data_size + 1 ) * sizeof (ap_Record);
  void * tmp = ap_resize_buffer( ap->data, min_size );
  if( !tmp ) return 0;
  ap->data = (ap_Record *)tmp;
  return &(ap->data[ap->data_size++]);
  }


static char push_back_option( Arg_parser * const ap, const int code,
                              const char * const long_name )
  {
  ap_Record * const p = push_back_record( ap );
  if( !p ) return 0;
  p->code = code;
  p->argument = 0;
  if( long_name )
    {
    const int len = strlen( long_name );
    p->parsed_name = (char *)malloc( len + 2 + 1 );
    if( !p->parsed_name ) return 0;
    p->parsed_name[0] = p->parsed_name[1] = '-';
    memcpy( p->parsed_name + 2, long_name, len + 1 );
    }
  else
    {
    p->parsed_name = (char *)malloc( 2 + 1 );
    if( !p->parsed_name ) return 0;
    p->parsed_name[0] = '-'; p->parsed_name[1] = code; p->parsed_name[2] = 0;
    }
  return 1;
  }


static char push_back_argument( Arg_parser * const ap,
                                const char * const argument )
  {
  ap_Record * const p = push_back_record( ap );
  if( !p ) return 0;
  p->code = 0;
  p->parsed_name = 0;
  return set_argument( ap, argument );
  }


static char set_error( Arg_parser * const ap, const char * const s1,
                       const char * const s2, const char * const s3 )
  {
  const int l1 = strlen( s1 ), l2 = strlen( s2 ), l3 = strlen( s3 );
  void * tmp = ap_resize_buffer( ap->error, l1 + l2 + l3 + 1 );
  if( !tmp ) return 0;
  ap->error = (char *)tmp;
  memcpy( ap->error, s1, l1 );
  memcpy( ap->error + l1, s2, l2 );
  memcpy( ap->error + l1 + l2, s3, l3 + 1 );
  return 1;
  }


static void free_data( Arg_parser * const ap )
  {
  int i;
  for( i = 0; i < ap->data_size; ++i )
    { free( ap->data[i].argument ); free( ap->data[i].parsed_name ); }
  if( ap->data ) { free( ap->data ); ap->data = 0; }
  ap->data_size = 0;
  }


/* Return 0 only if out of memory. */
static char parse_long_option( Arg_parser * const ap,
                               const char * const opt, const char * const arg,
                               const ap_Option options[], int * const argindp )
  {
  unsigned len;
  int index = -1, i;
  char exact = 0, ambig = 0;

  for( len = 0; opt[len+2] && opt[len+2] != '='; ++len ) ;

  /* Test all long options for either exact match or abbreviated matches. */
  for( i = 0; options[i].code != 0; ++i )
    if( options[i].long_name &&
        strncmp( options[i].long_name, &opt[2], len ) == 0 )
      {
      if( strlen( options[i].long_name ) == len )	/* Exact match found */
        { index = i; exact = 1; break; }
      else if( index < 0 ) index = i;		/* First nonexact match found */
      else if( options[index].code != options[i].code ||
               options[index].has_arg != options[i].has_arg )
        ambig = 1;		/* Second or later nonexact match found */
      }

  if( ambig && !exact )
    return set_error( ap, "option '", opt, "' is ambiguous" );

  if( index < 0 )		/* nothing found */
    return set_error( ap, "unrecognized option '", opt, "'" );

  ++*argindp;
  if( !push_back_option( ap, options[index].code, options[index].long_name ) )
    return 0;

  if( opt[len+2] )		/* '--<long_option>=<argument>' syntax */
    {
    if( options[index].has_arg == ap_no )
      return set_error( ap, "option '--", options[index].long_name,
                        "' doesn't allow an argument" );
    if( options[index].has_arg == ap_yes && !opt[len+3] )
      return set_error( ap, "option '--", options[index].long_name,
                        "' requires an argument" );
    return set_argument( ap, &opt[len+3] );	/* argument may be empty */
    }

  if( options[index].has_arg == ap_yes || options[index].has_arg == ap_yesme )
    {
    if( !arg || ( options[index].has_arg == ap_yes && !arg[0] ) )
      return set_error( ap, "option '--", options[index].long_name,
                        "' requires an argument" );
    ++*argindp;
    return set_argument( ap, arg );		/* argument may be empty */
    }

  return 1;
  }


/* Return 0 only if out of memory. */
static char parse_short_option( Arg_parser * const ap,
                                const char * const opt, const char * const arg,
                                const ap_Option options[], int * const argindp )
  {
  int cind = 1;			/* character index in opt */

  while( cind > 0 )
    {
    int index = -1, i;
    const unsigned char c = opt[cind];
    char code_str[2];
    code_str[0] = c; code_str[1] = 0;

    if( c != 0 )
      for( i = 0; options[i].code; ++i )
        if( c == options[i].code )
          { index = i; break; }

    if( index < 0 )
      return set_error( ap, "invalid option -- '", code_str, "'" );

    if( !push_back_option( ap, c, 0 ) ) return 0;
    if( opt[++cind] == 0 ) { ++*argindp; cind = 0; }	/* opt finished */

    if( options[index].has_arg != ap_no && cind > 0 && opt[cind] )
      {
      if( !set_argument( ap, &opt[cind] ) ) return 0;
      ++*argindp; cind = 0;
      }
    else if( options[index].has_arg == ap_yes || options[index].has_arg == ap_yesme )
      {
      if( !arg || ( options[index].has_arg == ap_yes && !arg[0] ) )
        return set_error( ap, "option requires an argument -- '", code_str, "'" );
      ++*argindp; cind = 0;
      if( !set_argument( ap, arg ) ) return 0;	/* argument may be empty */
      }
    }
  return 1;
  }


/* Return 0 only if out of memory. */
char ap_init( Arg_parser * const ap,
              const int argc, const char * const argv[],
              const ap_Option options[], const int flags )
  {
  const char ** non_options = 0;	/* skipped non-options */
  int non_options_size = 0;		/* number of skipped non-options */
  int argind = 1;			/* index in argv */
  char done = 0;			/* false until success or error */

  ap->data = 0;
  ap->error = 0;
  ap->data_size = 0;
  ap->argv_index = argc;
  if( argc < 2 || !argv || !options ) return 1;

  while( argind < argc )
    {
    const unsigned char ch1 = argv[argind][0];
    const unsigned char ch2 = ch1 ? argv[argind][1] : 0;

    if( ch1 == '-' && ch2 && ( ch2 == '-' || (flags & ap_neg_non_opt) == 0 ||
                               !is_number( argv[argind] + 1 ) ) )
      {
      const char * const opt = argv[argind];	/* we found an option */
      const char * const arg = ( argind + 1 < argc ) ? argv[argind+1] : 0;
      if( ch2 == '-' )
        {
        if( !argv[argind][2] ) { ++argind; break; }	/* we found "--" */
        else if( !parse_long_option( ap, opt, arg, options, &argind ) ) goto oom;
        }
      else if( !parse_short_option( ap, opt, arg, options, &argind ) ) goto oom;
      if( ap->error ) break;
      }
    else if( flags & (ap_in_order_stop | ap_in_order_skip) ) break;
    else if( flags & ap_in_order )
      { if( !push_back_argument( ap, argv[argind++] ) ) goto oom; }
    else
      {
      const int min_size = ( non_options_size + 1 ) * sizeof non_options[0];
      void * tmp = ap_resize_buffer( non_options, min_size );
      if( !tmp ) goto oom;
      non_options = (const char **)tmp;
      non_options[non_options_size++] = argv[argind++];
      }
    }
  if( ap->error ) free_data( ap );
  else if( flags & ap_in_order_skip ) ap->argv_index = argind;
  else						/* copy non-option arguments */
    {
    int i;
    for( i = 0; i < non_options_size; ++i )
      if( !push_back_argument( ap, non_options[i] ) ) goto oom;
    while( argind < argc )
      if( !push_back_argument( ap, argv[argind++] ) ) goto oom;
    }
  done = 1;
oom: if( non_options ) free( non_options );
  return done;
  }


void ap_free( Arg_parser * const ap )
  {
  free_data( ap );
  if( ap->error ) { free( ap->error ); ap->error = 0; }
  }


const char * ap_error( const Arg_parser * const ap ) { return ap->error; }

int ap_argv_index( const Arg_parser * const ap ) { return ap->argv_index; }

int ap_arguments( const Arg_parser * const ap ) { return ap->data_size; }

int ap_code( const Arg_parser * const ap, const int i )
  {
  if( i < 0 || i >= ap_arguments( ap ) ) return 0;
  return ap->data[i].code;
  }


const char * ap_parsed_name( const Arg_parser * const ap, const int i )
  {
  if( i < 0 || i >= ap_arguments( ap ) || !ap->data[i].parsed_name ) return "";
  return ap->data[i].parsed_name;
  }


const char * ap_argument( const Arg_parser * const ap, const int i )
  {
  if( i < 0 || i >= ap_arguments( ap ) || !ap->data[i].argument ) return "";
  return ap->data[i].argument;
  }
