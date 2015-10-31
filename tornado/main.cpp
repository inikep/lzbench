// Tornado - fast LZ77-based compression algorithm.
// This module is an example of command-line driver for the Tornado library.
//
// (c) Bulat Ziganshin
// (c) Joachim Henke
// This code is provided on the GPL license.
// If you need a commercial license to use the code, please write to Bulat.Ziganshin@gmail.com

#define FREEARC_STANDALONE_TORNADO
#include "Common.cpp"
#include "Tornado.cpp"

static const char *PROGRAM_NAME = "Tornado";

// Extensions of compressed and decompressed files
static const char *COMPRESS_EXT = ".tor", *DECOMPRESS_EXT = ".untor";

// Codec and parser names for humans
static const char *codec_name[]  = {"storing", "bytecoder", "bitcoder", "hufcoder", "aricoder"};
static const char *parser_name[] = {"", "greedy", "lazy", "flexible", "optimal"};

// Returns human-readable method description
static char *name (PackMethod method)
{
    static char namebuf[200], h[100], b[100], bt[100], bth[100], auxhash_size[100], u[100], ah[100], fb[100];
    const char*hashname[] = {"hash4", "???", "???", "???", "cchash4", "cchash5", "cchash6", "cchash7", "???", "???", "???", "???", "???", "???", "chash4", "chash5", "chash6", "chash7", "???", "???", "???", "???", "???", "???", "bt4", "bt5", "bt6", "bt7"};
    int c  = method.encoding_method;
    int l  = method.hash_row_width;
    showMem (method.hashsize,     h);
    showMem (method.buffer,       b);
    showMem (method.auxhash_size, auxhash_size);
    int x  = method.caching_finder;
    int p  = method.match_parser;
    int h3 = method.hash3;
    int hb  =  (x==NON_CACHING_MF? 4 : x%10);  // MINLEN for the primary match finder
    showMem64 (method.buffer*uint64(2)*sizeof(UINT), bt);
    sprintf (bth, (BT_MF4<=x && x<=BT_MF7)? "%s+":"", bt);
    sprintf (fb, p==OPTIMAL? " fb%d":"", method.fast_bytes);
    sprintf (u, !(CYCLED_MF4<=x && x<=CYCLED_MF7) && method.update_step!=999 && p!=OPTIMAL? "/u%d":"", method.update_step);  // Cycled MF ignore `step` in the update_hash()
    sprintf (ah, method.auxhash_size && hb>4? " + %s:%d %s":"", auxhash_size, method.auxhash_row_width, hb>5? (p==OPTIMAL? "chash4":"cchash4") : "exhash4");
    sprintf (namebuf, c==STORING? codec_name[c] : "%s parser%s, %s%s:%d%s %s%s%s, buffer %s, %s%s",
             parser_name[p], fb, bth, h, l, u, hashname[x], ah, h3==2?" + 256kb hash3 + 16kb hash2":h3?" + 64kb hash3 + 4kb hash2":"", b, codec_name[c], method.find_tables? "" : " w/o tables");
    return namebuf;
}

enum OPMODE {AUTO, _COMPRESS, _DECOMPRESS, BENCHMARK, HELP};

// Structure for recording compression statistics
struct Results {
  OPMODE mode;                 // Operation mode
  PackMethod method;           // Compression method used
  char method_name[100];       // Short name of compression method
  char *filename;              // Names of input/output files
  char outname[MY_FILENAME_MAX];
  FILE *fin, *fout;            // Input and output files
  FILESIZE filesize;           // Size of input file
  FILESIZE insize, outsize;    // How many bytes was already read/written
  FILESIZE progress_insize, progress_outsize;     // Bytes reported so far via the "progress" callback
  FILESIZE last_insize, last_outsize;             // Variables protecting the progress indicator from too often evaluation
  double start_time;           // When (de)compression was started
  double lasttime, lasttime2;  // Last time when we've updated progress indicator/console title
  bool   use_cpu_time;         // Compute pure CPU time used (instead of wall-clock time)
  bool   show_exact_percent;   // Show xx.x% progress indicator instead of xx%
  bool   quiet_title, quiet_header, quiet_progress, quiet_result;
                               // Don't show window title/compression header/progress/results
};

// Return current time (cpu time used or wall clock time). Return 0 if we disabled timing at compile time
#ifdef FREEARC_NO_TIMING
#define GetSomeTime() 0
#else
#define GetSomeTime() (r.use_cpu_time? GetThreadCPUTime() : GetGlobalTime())
#endif


// Temporary :) - until proper Unicode support will be implemented
#define delete_file(name)  (remove(name))
#define file_exists(name)  (access(name,0) == 0)
#define dir_exists(name)   (dir_exists0(name))
static inline int dir_exists0 (const char *name)
{
  struct _stat st;
  _stat(name,&st);
  return (st.st_mode & S_IFDIR) != 0;
}
int compress_all_at_once = 0;


// Print current compression statistics
static void start_print_stats (Results &r)
{
  r.insize = r.outsize = r.progress_insize = r.progress_outsize = r.last_insize = r.last_outsize = 0;
  r.show_exact_percent = FALSE;
  r.start_time = r.lasttime = r.lasttime2 = GetSomeTime();
#ifdef FREEARC_WIN
  if (strequ (r.filename, "-"))   // On windows, get_flen cannot return real filesize in situations like "type file|tor"
      r.filesize = -1;
  else
#endif
      r.filesize = get_flen(r.fin);
  sprintf(r.method_name, r.mode==_COMPRESS? "-%d: " : "", r.method.number);
}

// Print current compression statistics
static void print_stats (Results &r)
{
#ifndef FREEARC_NO_TIMING
  FILESIZE insize  = r.progress_insize;   bool not_enough_input   =  (insize  - r.last_insize  < PROGRESS_CHUNK_SIZE);
  FILESIZE outsize = r.progress_outsize;  bool not_enough_output  =  (outsize - r.last_outsize < PROGRESS_CHUNK_SIZE);
  if (r.quiet_progress  ||  insize==0  ||  outsize==0  ||  not_enough_input && not_enough_output)  return;   // Prints stats no more often than once per 64 kb
  r.last_insize  = insize;
  r.last_outsize = outsize;
  double curtime = GetSomeTime();

  // Update progress indicator every 0.2 seconds
  if (curtime > r.lasttime+0.2)
  {
    double time = curtime - r.start_time;      // Time used so far
    char percents0[100] = "",  remains0[100] = "", percents[1000] = "",  remains[100] = "",  insize_str[100],  outsize_str[100];
    if (r.filesize)
    {
      // If progress by 1% takes more than 1-2 seconds - display xx.x%, otherwise xx%
      // (we don't want to switch it too frequently, therefore "?1:2")
      r.show_exact_percent  =  double(r.filesize)/insize*time > (r.show_exact_percent?1:2)*100;
      if (r.show_exact_percent)
           sprintf (percents0, "%.1lf%%", double(int(double(insize)*100/r.filesize*10))/10);
      else sprintf (percents0, "%d%%", int(double(insize)*100/r.filesize));
      sprintf (percents, "%s: ", percents0);

      int remain = int(double(r.filesize-insize)/insize*time)+1;
      if (remain>=3600)
           sprintf (remains0, "%02d:%02d:%02d", remain / 3600, (remain % 3600) / 60, remain % 60);
      else sprintf (remains0, "%02d:%02d", remain / 60, remain % 60);
      sprintf (remains, ". Remains %s", remains0);
    }
    double origsize  = (r.mode==_COMPRESS? insize  : outsize);         // Size of original (uncompressed) data
    double compsize  = (r.mode==_COMPRESS? outsize : insize);          // Size of compressed data
    double ratio     = (compsize/origsize)*100;
    double speed     = (origsize/mb) / mymax(time,0.001);              // Speed of processing, in MiB/s
    if (!r.quiet_result && !strequ (r.outname, "-"))
      fprintf (stderr, "\r%s%s%s -> %s: %.2lf%%, speed %.3lf mb/sec%s   ",
               r.method_name, percents, show3(insize,insize_str), show3(outsize,outsize_str), ratio, speed, remains);

    if (!r.quiet_title  &&  r.filesize  &&  curtime > r.lasttime2+0.5)   // Update window title every 0.5 seconds
    {
      const char *op  =  (r.mode==_COMPRESS? "Compressing": r.mode==_DECOMPRESS? "Extracting": "Benchmarking");
      sprintf (percents, "%s %s | %s %s", percents0, remains0, op, strequ(r.filename,"-")? PROGRAM_NAME : r.filename);
      EnvSetConsoleTitleA (percents);
      Taskbar_SetProgressValue (insize, r.filesize);
      r.lasttime2 = curtime;
    }
    r.lasttime = curtime;
  }
#endif
}

static void clear_window_title()
{
#ifndef FREEARC_NO_TIMING
  Taskbar_Done();
  EnvResetConsoleTitle();
#endif
}

static void signal_handler(int)
{
  clear_window_title();
  fprintf (stderr, "\nERROR: ^Break pressed\n");
  exit(FREEARC_EXIT_ERROR);
}


// Print final compression statistics
static void print_final_stats (Results &r)
{
  if (!r.quiet_result && r.insize && r.outsize)
  {
    double origsize  = (r.mode==_COMPRESS? r.insize  : r.outsize);         // Size of original (uncompressed) data
    double compsize  = (r.mode==_COMPRESS? r.outsize : r.insize);          // Size of compressed data
    double ratio     = (compsize/origsize)*100;
    char insize_str[100],  outsize_str[100];
    fprintf (stderr, "\r%s%s %s -> %s: %.2lf%%", r.method_name, r.mode==_COMPRESS? "compressed":"Unpacked",
             show3(r.insize,insize_str), show3(r.outsize,outsize_str), ratio);
#ifndef FREEARC_NO_TIMING
    double time = GetSomeTime() - r.start_time;                            // Time spent for (de)compression
    if (time > 0.001) {
      double speed = (origsize/mb) / time;                                 // Speed of processing, in MiB/s
      fprintf (stderr, ", time %.3lf secs, speed %.3lf mb/sec", time, speed);
    }
#endif
    fprintf (stderr, "\n");
  }
  if (!r.quiet_title && r.filesize)
    clear_window_title();
}


// Callback function called by compression routine to read/write data.
static int ReadWriteCallback (const char *what, void *buf, int size, void *_r)
{
  Results &r = *(Results*)_r;        // Accumulator for compression statistics

  if (strequ(what,"read")) {
    int n = file_read (r.fin, buf, size);
    r.insize += n;
    return n;

  } else if (strequ(what,"write")) {
    if (r.fout)
      if (size != file_write (r.fout, buf, size))
        return FREEARC_ERRCODE_WRITE;
    r.outsize += size;
    return size;

  } else if (strequ(what,"progress")) {
    r.progress_insize  += ((int64*)buf)[0];
    r.progress_outsize += ((int64*)buf)[1];
    print_stats(r);
    return FREEARC_OK;

  } else {
    return FREEARC_ERRCODE_NOT_IMPLEMENTED;
  }
}


// ****************************************************************************************************************************
// Checking option values *****************************************************************************************************
// ****************************************************************************************************************************

const int64 MIN_BUFFER_SIZE = 4*kb,  MAX_BUFFER_SIZE = 1*gb,
            MIN_HASH_SIZE = 4*kb,  MAX_HASH_SIZE = UINT_MAX,
            MAX_HASH_ROW_WIDTH = 64*kb,  MAX_UPDATE_STEP = 64*kb,
            MIN_FAST_BYTES = 1,  MAX_FAST_BYTES = 64*kb;

static int64 check_int (int64 x, int error, int64 minVal, int64 maxVal, char *option)
{
  if (error) {
    fprintf (stderr, "ERROR: bad option format: '%s'\n", option);
    exit(FREEARC_EXIT_ERROR);
  }
  if (x<minVal || x>maxVal) {
    char minStr[100], maxStr[100];
    showMem64(minVal,minStr);
    showMem64(maxVal,maxStr);
    fprintf (stderr, "ERROR: option '%s' value should be in the %s..%s range\n", option, minStr, maxStr);
    exit(FREEARC_EXIT_ERROR);
  }
  return x;
}

static int64 check_parse_int (char *param, int64 minVal, int64 maxVal, char *option)
{
  int error = 0;  int x = parseInt (param, &error);
  return check_int (x, error, minVal, maxVal, option);
}

static int64 check_parse_mem (char *param, int64 minVal, int64 maxVal, char *option)
{
  int error = 0;  int64 x = parseMem64 (param, &error);
  return check_int (x, error, minVal, maxVal, option);
}


int main (int argc, char **argv)
{
    // Optimize allocation for Tornado (2GB hash allocated after 1GB dictionary)
    AllocTopDown = FALSE;

    // Operation mode
    OPMODE global_mode=AUTO;

    // Record that stores all the info required for ReadWriteCallback
    static Results r;
    r.use_cpu_time = r.quiet_title = r.quiet_header = r.quiet_progress = r.quiet_result = FALSE;

    // Default compression parameters are equivalent to option -5
    r.method = std_Tornado_method [default_Tornado_method];

    // Delete successfully (de)compressed input files
    bool delete_input_files = FALSE;

    // Count of files to process
    int fcount=0;

    // Output path/filename
    const char *output_filename = NULL;

    // Process options until "--"
    // 1. First, process -1..-16 option if any
    for (char **argv_ptr = argv; *++argv_ptr!=NULL; ) {
        char *param = *argv_ptr;
        if (*param == '-') {
            param++;
                 if (strcasecmp(param,"-")==0)   break;   // "--" is a "stop processing" option
            else if (isdigit(*param))            r.method = std_Tornado_method[check_parse_int (param, 1, elements(std_Tornado_method)-1, param-1)];
        }
    }
    // 2. Second, process rest of options
    for (char **argv_ptr = argv; *++argv_ptr!=NULL; ) {
        char *param = *argv_ptr;
        if (param[0] != '-' || param[1]=='\0') {
            fcount++;
        } else { param++;  int error=0;
                 if (strcasecmp(param,"-")==0)      break;
            else if (strcasecmp(param,"") ==0)      continue;
            else if (strcasecmp(param,"z")==0)      global_mode=_COMPRESS;
            else if (strcasecmp(param,"d")==0)      global_mode=_DECOMPRESS;
            else if (strcasecmp(param,"delete")==0) delete_input_files=TRUE;
            else if (strcasecmp(param,"t")==0)      output_filename="";
            else if (strcasecmp(param,"q")==0)      r.quiet_title = r.quiet_header = r.quiet_progress = r.quiet_result = TRUE;
#ifndef FREEARC_NO_TIMING
            else if (strcasecmp(param,"cpu")==0)    r.use_cpu_time=TRUE;
#endif
            else if (strcasecmp(param,"h")==0)      global_mode=HELP;
            else if (strcasecmp(param,"b")==0)      r.method.buffer = MAX_BUFFER_SIZE;         // set buffer size to the maximum supported by LZ coders
            else if (strcasecmp(param,"x")==0)      r.method.caching_finder = CACHING_MF4;
            else if (strcasecmp(param,"xx")==0)     r.method.caching_finder = CYCLED_MF4;
            else if (strcasecmp(param,"x+")==0)     r.method.caching_finder = CACHING_MF4;
            else if (strcasecmp(param,"x-")==0)     r.method.caching_finder = NON_CACHING_MF;
            else if (strcasecmp(param,"t+")==0)     r.method.find_tables = TRUE;
            else if (strcasecmp(param,"t-")==0)     r.method.find_tables = FALSE;
            else if (strcasecmp(param,"t1")==0)     r.method.find_tables = TRUE;
            else if (strcasecmp(param,"t0")==0)     r.method.find_tables = FALSE;
            else if (strcasecmp(param,"s")==0)      r.method.hash3 = 1;
            else if (strcasecmp(param,"ss")==0)     r.method.hash3 = 2;
            else if (strcasecmp(param,"s+")==0)     r.method.hash3 = 1;
            else if (strcasecmp(param,"s-")==0)     r.method.hash3 = 0;
            else if (start_with(param,"fb"))        r.method.fast_bytes = check_parse_int (param+2, MIN_FAST_BYTES, MAX_FAST_BYTES, *argv_ptr);
#ifdef FREEARC_WIN
            else if (strcasecmp(param,"slp-")==0)   DefaultLargePageMode = DISABLE;
            else if (strcasecmp(param,"slp" )==0)   DefaultLargePageMode = TRY;
            else if (strcasecmp(param,"slp+")==0)   DefaultLargePageMode = FORCE;
#endif
            else if (start_with(param,"rem"))       /* ignore option */;
            else if (isdigit(*param))            ; // -1..-16 option is already processed :)
            else switch( tolower(*param++) ) {
                case 'b': r.method.buffer          = check_parse_mem (param, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE,    *argv_ptr);  break;
                case 'h': r.method.hashsize        = check_parse_mem (param, MIN_HASH_SIZE,   MAX_HASH_SIZE,      *argv_ptr);  break;
                case 'l': r.method.hash_row_width  = check_parse_int (param, 1,               MAX_HASH_ROW_WIDTH, *argv_ptr);  break;
                case 'u': r.method.update_step     = check_parse_int (param, 1,               MAX_UPDATE_STEP,    *argv_ptr);  break;
                case 'c': r.method.encoding_method = check_parse_int (param, BYTECODER,       ARICODER,           *argv_ptr);  break;
                case 's': r.method.hash3           = check_parse_int (param, 0,               2,                  *argv_ptr);  break;
                case 'p': r.method.match_parser    = parseInt (param, &error);  break;
                case 'x': r.method.caching_finder  = parseInt (param, &error);  break;
                case 'o': output_filename          = param;                     break;
                case 'q':
#ifndef FREEARC_NO_TIMING
                          r.quiet_title            = strchr (param, 't');
                          r.quiet_progress         = strchr (param, 'p');
#endif
                          r.quiet_header           = strchr (param, 'h');
                          r.quiet_result           = strchr (param, 'r');
                          break;
                case 'a': switch( tolower(*param++) ) {
                            case 'h': r.method.auxhash_size      = check_parse_mem (param, MIN_HASH_SIZE, MAX_HASH_SIZE, *argv_ptr);  goto check_for_errors;
                            case 'l': r.method.auxhash_row_width = check_parse_int (param, 1,        MAX_HASH_ROW_WIDTH, *argv_ptr);  goto check_for_errors;
                          }
                          // 'a' should be the last case!
                default : fprintf (stderr, "ERROR: unknown option '%s'\n", *argv_ptr);
                          exit(FREEARC_EXIT_ERROR);
            }
check_for_errors:
            if (error) {
                fprintf (stderr, "ERROR: bad option format: '%s'\n", *argv_ptr);
                exit(FREEARC_EXIT_ERROR);
            }
            if (!(r.method.match_parser==GREEDY || r.method.match_parser==LAZY  || r.method.match_parser==OPTIMAL)) {
                fprintf (stderr, "ERROR: bad option value: '%s'\n", *argv_ptr);
                exit(FREEARC_EXIT_ERROR);
            }
            int mf = r.method.caching_finder;
            r.method.caching_finder = mf = (mf==CACHING_MF4_DUB? CACHING_MF4 : (mf==CYCLED_MF4_DUB? CYCLED_MF4 : mf));
            if (!(mf==NON_CACHING_MF || (CYCLED_MF4<=mf && mf<=CYCLED_MF7) || (CACHING_MF4<=mf && mf<=CACHING_MF7) || (BT_MF4<=mf && mf<=BT_MF7))) {
                fprintf (stderr, "ERROR: non-existing match finder: '%s'\n", *argv_ptr);
                exit(FREEARC_EXIT_ERROR);
            }
        }
    }

    // No files to compress: read from stdin and write to stdout
    if (global_mode!=HELP && fcount==0 &&
       (global_mode!=AUTO  ||  !isatty(0) && !isatty(1)) ) {

        static char *_argv[] = {argv[0], (char*)"-", NULL};
        argv = _argv;
        fcount = 1;

    } else if (global_mode==HELP || fcount==0) {
        char h[100], ah[100], b[100], MinHashSizeStr[100], MaxHashSizeStr[100], MinBufSizeStr[100], MaxBufSizeStr[100];
        showMem64 (r.method.hashsize, h);
        showMem64 (r.method.auxhash_size, ah);
        showMem64 (r.method.buffer, b);
        showMem64 (MIN_HASH_SIZE, MinHashSizeStr);
        showMem64 (MAX_HASH_SIZE+1, MaxHashSizeStr);
        showMem64 (MIN_BUFFER_SIZE, MinBufSizeStr);
        showMem64 (MAX_BUFFER_SIZE, MaxBufSizeStr);
        printf( "Tornado compressor v0.6  (c) Bulat.Ziganshin@gmail.com  http://freearc.org  2014-03-08\n"
                "\n"
                " Usage: tor [options and files in any order]\n"
                "   -#      -- compression level (1..%d), default %d\n", int(elements(std_Tornado_method))-1, default_Tornado_method);
        printf( "   -z      -- force compression\n"
                "   -d      -- force decompression\n"
                "   -oNAME  -- output filename/directory (default %s/%s)\n", COMPRESS_EXT, DECOMPRESS_EXT);
        printf( "   -t      -- test (de)compression (redirect output to nul)\n"
                "   -delete -- delete successfully (de)compressed input files\n"
#ifdef FREEARC_NO_TIMING
                "   -q      -- be quiet; -q[hr]* disables header/results individually\n"
#else
                "   -q      -- be quiet; -q[thpr]* disables title/header/progress/results individually\n"
                "   -cpu    -- compute raw CPU time (for benchmarking)\n"
#endif
#ifdef FREEARC_WIN
                "   -slp[+/-/]   -- force/disable/try(default) large pages support (2mb/4mb)\n"
#endif
                "   -rem... -- command-line remark\n"
                "   -h      -- display this help\n"
                "   --      -- stop flags processing\n"
                " \"-\" used as filename means stdin/stdout\n"
                "\n"
                " Advanced compression parameters:\n"
                "   -b#     -- buffer size (%s..%s), default %s\n", MinBufSizeStr, MaxBufSizeStr, b);
        printf( "   -h#     -- hash size (%s..%s-1), default %s\n", MinHashSizeStr, MaxHashSizeStr, h);
        printf( "   -l#     -- length of hash row (1..%d), default %d\n", int(MAX_HASH_ROW_WIDTH), r.method.hash_row_width);
        printf( "   -ah#    -- auxiliary hash size (%s..%s-1), default %s\n", MinHashSizeStr, MaxHashSizeStr, ah);
        printf( "   -al#    -- auxiliary hash row length (1..%d), default %d\n", int(MAX_HASH_ROW_WIDTH), r.method.auxhash_row_width);
        printf( "   -u#     -- update step (1..%d), default %d\n", int(MAX_UPDATE_STEP), r.method.update_step);
        printf( "   -c#     -- coder (1-bytes, 2-bits, 3-huf, 4-arith), default %d\n", r.method.encoding_method);
        printf( "   -p#     -- parser (1-greedy, 2-lazy, 4-optimal), default %d\n", r.method.match_parser);
        printf( "   -x#     -- match finder (0: non-caching ht4, 4-7: cycling cached ht4..ht7,\n"
                "                            14-17: shifting cached ht4..ht7, 24-27: bt4..bt7), default %d\n", r.method.caching_finder);
        printf( "   -s#     -- 2/3-byte hash (0-disabled, 1-fast, 2-max), default %d\n", r.method.hash3);
        printf( "   -t#     -- table diffing (0-disabled, 1-enabled), default %d\n", r.method.find_tables);
        printf( "   -fb#    -- fast bytes in the optimal parser (%d..%d), default %d\n", int(MIN_FAST_BYTES), int(MAX_FAST_BYTES), r.method.fast_bytes);
        printf( "\n"
                " Predefined methods:\n");
        for (int i=1; i<elements(std_Tornado_method); i++)
        {
            printf("   %-8d-- %s\n", -i, name(std_Tornado_method[i]));
        }
        exit(EXIT_SUCCESS);
    }



    // (De)compress all files given on cmdline
    bool parse_options=TRUE;  // options will be parsed until "--"
    Install_signal_handler(signal_handler);

    for (char **parameters = argv; *++parameters!=NULL; )
    {
        // If options are still parsed and this argument starts with "-" - it's an option
        if (parse_options && parameters[0][0]=='-' && parameters[0][1]) {
            if (strequ(*parameters,"--"))  parse_options=FALSE;
            continue;
        }

        // Save input filename
        r.filename = *parameters;

        // Select operation mode if it was not specified on cmdline
        r.mode = global_mode != AUTO?                  global_mode :
                 end_with (r.filename, COMPRESS_EXT)?  _DECOMPRESS  :
                                                       _COMPRESS;
        // Extension that should be added to output filenames
        const char *MODE_EXT  =  r.mode==_COMPRESS? COMPRESS_EXT : DECOMPRESS_EXT;

        // Construct output filename
        if (r.mode==BENCHMARK  ||  output_filename && strequ (output_filename, "")) {  // Redirect output to nul
            strcpy (r.outname, "");
        } else if (output_filename) {
            if (strequ(output_filename,"-"))
                strcpy (r.outname, output_filename);
            else if (is_path_char (last_char (output_filename)))
                {sprintf(r.outname, "%s%s", output_filename, drop_dirname(r.filename));  goto add_remove_ext;}
            else if (dir_exists (output_filename))
                {sprintf(r.outname, "%s%c%s", output_filename, PATH_DELIMITER, drop_dirname(r.filename));  goto add_remove_ext;}
            else
                strcpy (r.outname, output_filename);
        } else if (strequ(r.filename,"-")) {
            strcpy (r.outname, r.filename);
        } else {
            // No output filename was given on cmdline:
            //    on compression   - add COMPRESS_EXT
            //    on decompression - remove COMPRESS_EXT (and add DECOMPRESS_EXT if file already exists)
            strcpy (r.outname, r.filename);
add_remove_ext: // Remove COMPRESS_EXT on the end of name or add DECOMPRESS_EXT (unless we are in _COMPRESS mode)
            if (r.mode!=_COMPRESS && end_with (r.outname, COMPRESS_EXT)) {
                r.outname [strlen(r.outname) - strlen(COMPRESS_EXT)] = '\0';
                if (file_exists (r.outname))
                    strcat(r.outname, MODE_EXT);
            } else {
                strcat(r.outname, MODE_EXT);
            }
        }

        // Open input file
        r.fin = strequ (r.filename, "-")? stdin : file_open_read_binary(r.filename);
        if (r.fin == NULL) {
            fprintf (stderr, "\n Can't open %s for read\n", r.filename);
            exit(FREEARC_EXIT_ERROR);
        }
        set_binary_mode (r.fin);

        // Open output file
        if (*r.outname) {
          r.fout = strequ (r.outname, "-")? stdout : file_open_write_binary(r.outname);
          if (r.fout == NULL) {
              fprintf (stderr, "\n Can't open %s for write\n", r.outname);
              exit(FREEARC_EXIT_ERROR);
          }
          set_binary_mode (r.fout);
        } else {
          r.fout = NULL;
        }

        // Prepare to (de)compression
        int result;  char filesize_str[100];
        start_print_stats(r);

        // Perform actual (de)compression
        switch (r.mode) {
        case _COMPRESS: {
            if (!r.quiet_header && r.filesize >= 0)
                fprintf (stderr, "Compressing %s bytes with %s\n", show3(r.filesize,filesize_str), name(r.method));
            PackMethod m = r.method;
            if (r.filesize >= 0)
                m.buffer = mymin (m.buffer, r.filesize+LOOKAHEAD*2);
            result = tor_compress (m, ReadWriteCallback, &r, NULL, -1);
            break; }

        case _DECOMPRESS: {
            //if (!r.quiet_header && !strequ (r.outname, "-"))   fprintf (stderr, "Unpacking %.3lf mb\n", double(r.filesize)/1000/1000);
            result = tor_decompress (ReadWriteCallback, &r, NULL, -1);
            break; }
        }

        // Finish (de)compression
        print_final_stats(r);
        fclose (r.fin);
        if (r.fout)  fclose (r.fout);

        if (result == FREEARC_OK)  {
            if (delete_input_files && !strequ(r.filename,"-"))    delete_file(r.filename);
        } else {
            if (!strequ(r.outname,"-") && !strequ(r.outname,""))  delete_file(r.outname);
            switch (result) {
            case FREEARC_ERRCODE_INVALID_COMPRESSOR:
                fprintf (stderr, "\nThis compression mode isn't supported by small Tornado version, use full version instead!");
                break;
            case FREEARC_ERRCODE_NOT_ENOUGH_MEMORY:
                fprintf (stderr, "\nNot enough memory for (de)compression!");
                break;
            case FREEARC_ERRCODE_READ:
                fprintf (stderr, "\nRead error! Bad media?");
                break;
            case FREEARC_ERRCODE_WRITE:
                fprintf (stderr, "\nWrite error! Disk full?");
                break;
            case FREEARC_ERRCODE_BAD_COMPRESSED_DATA:
                fprintf (stderr, "\nData can't be decompressed!");
                break;
            default:
                fprintf (stderr, "\n(De)compression failed with error code %d!", result);
                break;
            }
            exit(FREEARC_EXIT_ERROR);
        }

        // going to next file...
    }

    return EXIT_SUCCESS;
}
