// zpaq.cpp - Journaling incremental deduplicating archiver

#define ZPAQ_VERSION "7.15"
/*
  This software is provided as-is, with no warranty.
  I, Matt Mahoney, release this software into
  the public domain.   This applies worldwide.
  In some countries this may not be legally possible; if so:
  I grant anyone the right to use this software for any purpose,
  without any conditions, unless such conditions are required by law.

zpaq is a journaling (append-only) archiver for incremental backups.
Files are added only when the last-modified date has changed. Both the old
and new versions are saved. You can extract from old versions of the
archive by specifying a date or version number. zpaq supports 5
compression levels, deduplication, AES-256 encryption, and multi-threading
using an open, self-describing format for backward and forward
compatibility in Windows and Linux. See zpaq.pod for usage.

TO COMPILE:

This program needs libzpaq from http://mattmahoney.net/zpaq/
Recommended compile for Windows with MinGW:

  g++ -O3 zpaq.cpp libzpaq.cpp -o zpaq

With Visual C++:

  cl /O2 /EHsc zpaq.cpp libzpaq.cpp advapi32.lib

For Linux:

  g++ -O3 -Dunix zpaq.cpp libzpaq.cpp -pthread -o zpaq

For BSD or OS/X

  g++ -O3 -Dunix -DBSD zpaq.cpp libzpaq.cpp -pthread -o zpaq

Possible options:

  -DDEBUG    Enable run time checks and help screen for undocumented options.
  -DNOJIT    Don't assume x86 with SSE2 for libzpaq. Slower (disables JIT).
  -Dunix     Not Windows. Sometimes automatic in Linux. Needed for Mac OS/X.
  -DBSD      For BSD or OS/X.
  -DPTHREAD  Use Pthreads instead of Windows threads. Requires pthreadGC2.dll
             or pthreadVC2.dll from http://sourceware.org/pthreads-win32/
  -Dunixtest To make -Dunix work in Windows with MinGW.
  -fopenmp   Parallel divsufsort (faster, implies -pthread, broken in MinGW).
  -pthread   Required in Linux, implied by -fopenmp.
  -O3 or /O2 Optimize (faster).
  -o         Name of output executable.
  /EHsc      Enable exception handing in VC++ (required).
  advapi32.lib  Required for libzpaq in VC++.

*/
#define _FILE_OFFSET_BITS 64  // In Linux make sizeof(off_t) == 8
#ifndef UNICODE
#define UNICODE  // For Windows
#endif
#include "libzpaq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <fcntl.h>

#ifndef DEBUG
#define NDEBUG 1
#endif
#include <assert.h>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#ifndef unix
#define unix 1
#endif
#endif
#ifdef unix
#define PTHREAD 1
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#include <utime.h>
#include <errno.h>
#ifdef BSD
#include <sys/sysctl.h>
#endif

#else  // Assume Windows
#include <windows.h>
#include <io.h>
#endif

// For testing -Dunix in Windows
#ifdef unixtest
#define lstat(a,b) stat(a,b)
#define mkdir(a,b) mkdir(a)
#ifndef fseeko
#define fseeko(a,b,c) fseeko64(a,b,c)
#endif
#ifndef ftello
#define ftello(a) ftello64(a)
#endif
#endif

using std::string;
using std::vector;
using std::map;
using std::min;
using std::max;
using libzpaq::StringBuffer;

// Handle errors in libzpaq and elsewhere
void libzpaq::error(const char* msg) {
  if (strstr(msg, "ut of memory")) throw std::bad_alloc();
  throw std::runtime_error(msg);
}
using libzpaq::error;

// Portable thread types and functions for Windows and Linux. Use like this:
//
// // Create mutex for locking thread-unsafe code
// Mutex mutex;            // shared by all threads
// init_mutex(mutex);      // initialize in unlocked state
// Semaphore sem(n);       // n >= 0 is initial state
//
// // Declare a thread function
// ThreadReturn thread(void *arg) {  // arg points to in/out parameters
//   lock(mutex);          // wait if another thread has it first
//   release(mutex);       // allow another waiting thread to continue
//   sem.wait();           // wait until n>0, then --n
//   sem.signal();         // ++n to allow waiting threads to continue
//   return 0;             // must return 0 to exit thread
// }
//
// // Start a thread
// ThreadID tid;
// run(tid, thread, &arg); // runs in parallel
// join(tid);              // wait for thread to return
// destroy_mutex(mutex);   // deallocate resources used by mutex
// sem.destroy();          // deallocate resources used by semaphore

#ifdef PTHREAD
#include <pthread.h>
typedef void* ThreadReturn;                                // job return type
typedef pthread_t ThreadID;                                // job ID type
void run(ThreadID& tid, ThreadReturn(*f)(void*), void* arg)// start job
  {pthread_create(&tid, NULL, f, arg);}
void join(ThreadID tid) {pthread_join(tid, NULL);}         // wait for job
typedef pthread_mutex_t Mutex;                             // mutex type
void init_mutex(Mutex& m) {pthread_mutex_init(&m, 0);}     // init mutex
void lock(Mutex& m) {pthread_mutex_lock(&m);}              // wait for mutex
void release(Mutex& m) {pthread_mutex_unlock(&m);}         // release mutex
void destroy_mutex(Mutex& m) {pthread_mutex_destroy(&m);}  // destroy mutex

class Semaphore {
public:
  Semaphore() {sem=-1;}
  void init(int n) {
    assert(n>=0);
    assert(sem==-1);
    pthread_cond_init(&cv, 0);
    pthread_mutex_init(&mutex, 0);
    sem=n;
  }
  void destroy() {
    assert(sem>=0);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cv);
  }
  int wait() {
    assert(sem>=0);
    pthread_mutex_lock(&mutex);
    int r=0;
    if (sem==0) r=pthread_cond_wait(&cv, &mutex);
    assert(sem>0);
    --sem;
    pthread_mutex_unlock(&mutex);
    return r;
  }
  void signal() {
    assert(sem>=0);
    pthread_mutex_lock(&mutex);
    ++sem;
    pthread_cond_signal(&cv);
    pthread_mutex_unlock(&mutex);
  }
private:
  pthread_cond_t cv;  // to signal FINISHED
  pthread_mutex_t mutex; // protects cv
  int sem;  // semaphore count
};

#else  // Windows
typedef DWORD ThreadReturn;
typedef HANDLE ThreadID;
void run(ThreadID& tid, ThreadReturn(*f)(void*), void* arg) {
  tid=CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)f, arg, 0, NULL);
  if (tid==NULL) error("CreateThread failed");
}
void join(ThreadID& tid) {WaitForSingleObject(tid, INFINITE);}
typedef HANDLE Mutex;
void init_mutex(Mutex& m) {m=CreateMutex(NULL, FALSE, NULL);}
void lock(Mutex& m) {WaitForSingleObject(m, INFINITE);}
void release(Mutex& m) {ReleaseMutex(m);}
void destroy_mutex(Mutex& m) {CloseHandle(m);}

class Semaphore {
public:
  enum {MAXCOUNT=2000000000};
  Semaphore(): h(NULL) {}
  void init(int n) {assert(!h); h=CreateSemaphore(NULL, n, MAXCOUNT, NULL);}
  void destroy() {assert(h); CloseHandle(h);}
  int wait() {assert(h); return WaitForSingleObject(h, INFINITE);}
  void signal() {assert(h); ReleaseSemaphore(h, 1, NULL);}
private:
  HANDLE h;  // Windows semaphore
};

#endif

// Global variables
int64_t global_start=0;  // set to mtime() at start of main()

// In Windows, convert 16-bit wide string to UTF-8 and \ to /
#ifndef unix
string wtou(const wchar_t* s) {
  assert(sizeof(wchar_t)==2);  // Not true in Linux
  assert((wchar_t)(-1)==65535);
  string r;
  if (!s) return r;
  for (; *s; ++s) {
    if (*s=='\\') r+='/';
    else if (*s<128) r+=*s;
    else if (*s<2048) r+=192+*s/64, r+=128+*s%64;
    else r+=224+*s/4096, r+=128+*s/64%64, r+=128+*s%64;
  }
  return r;
}

// In Windows, convert UTF-8 string to wide string ignoring
// invalid UTF-8 or >64K. Convert "/" to slash (default "\").
std::wstring utow(const char* ss, char slash='\\') {
  assert(sizeof(wchar_t)==2);
  assert((wchar_t)(-1)==65535);
  std::wstring r;
  if (!ss) return r;
  const unsigned char* s=(const unsigned char*)ss;
  for (; s && *s; ++s) {
    if (s[0]=='/') r+=slash;
    else if (s[0]<128) r+=s[0];
    else if (s[0]>=192 && s[0]<224 && s[1]>=128 && s[1]<192)
      r+=(s[0]-192)*64+s[1]-128, ++s;
    else if (s[0]>=224 && s[0]<240 && s[1]>=128 && s[1]<192
             && s[2]>=128 && s[2]<192)
      r+=(s[0]-224)*4096+(s[1]-128)*64+s[2]-128, s+=2;
  }
  return r;
}
#endif

// Print a UTF-8 string to f (stdout, stderr) so it displays properly
void printUTF8(const char* s, FILE* f=stdout) {
  assert(f);
  assert(s);
#ifdef unix
  fprintf(f, "%s", s);
#else
  const HANDLE h=(HANDLE)_get_osfhandle(_fileno(f));
  DWORD ft=GetFileType(h);
  if (ft==FILE_TYPE_CHAR) {
    fflush(f);
    std::wstring w=utow(s, '/');  // Windows console: convert to UTF-16
    DWORD n=0;
    WriteConsole(h, w.c_str(), w.size(), &n, 0);
  }
  else  // stdout redirected to file
    fprintf(f, "%s", s);
#endif
}

// Return relative time in milliseconds
int64_t mtime() {
#ifdef unix
  timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec*1000LL+tv.tv_usec/1000;
#else
  int64_t t=GetTickCount();
  if (t<global_start) t+=0x100000000LL;
  return t;
#endif
}

// Convert 64 bit decimal YYYYMMDDHHMMSS to "YYYY-MM-DD HH:MM:SS"
// where -1 = unknown date, 0 = deleted.
string dateToString(int64_t date) {
  if (date<=0) return "                   ";
  string s="0000-00-00 00:00:00";
  static const int t[]={18,17,15,14,12,11,9,8,6,5,3,2,1,0};
  for (int i=0; i<14; ++i) s[t[i]]+=int(date%10), date/=10;
  return s;
}

// Convert attributes to a readable format
string attrToString(int64_t attrib) {
  string r="     ";
  if ((attrib&255)=='u') {
    r[0]="0pc3d5b7 9lBsDEF"[(attrib>>20)&15];
    for (int i=0; i<4; ++i)
      r[4-i]=(attrib>>(8+3*i))%8+'0';
  }
  else if ((attrib&255)=='w') {
    for (int i=0, j=0; i<32; ++i) {
      if ((attrib>>(i+8))&1) {
        char c="RHS DAdFTprCoIEivs89012345678901"[i];
        if (j<5) r[j]=c;
        else r+=c;
        ++j;
      }
    }
  }
  return r;
}

// Convert seconds since 0000 1/1/1970 to 64 bit decimal YYYYMMDDHHMMSS
// Valid from 1970 to 2099.
int64_t decimal_time(time_t tt) {
  if (tt==-1) tt=0;
  int64_t t=(sizeof(tt)==4) ? unsigned(tt) : tt;
  const int second=t%60;
  const int minute=t/60%60;
  const int hour=t/3600%24;
  t/=86400;  // days since Jan 1 1970
  const int term=t/1461;  // 4 year terms since 1970
  t%=1461;
  t+=(t>=59);  // insert Feb 29 on non leap years
  t+=(t>=425);
  t+=(t>=1157);
  const int year=term*4+t/366+1970;  // actual year
  t%=366;
  t+=(t>=60)*2;  // make Feb. 31 days
  t+=(t>=123);   // insert Apr 31
  t+=(t>=185);   // insert June 31
  t+=(t>=278);   // insert Sept 31
  t+=(t>=340);   // insert Nov 31
  const int month=t/31+1;
  const int day=t%31+1;
  return year*10000000000LL+month*100000000+day*1000000
         +hour*10000+minute*100+second;
}

// Convert decimal date to time_t - inverse of decimal_time()
time_t unix_time(int64_t date) {
  if (date<=0) return -1;
  static const int days[12]={0,31,59,90,120,151,181,212,243,273,304,334};
  const int year=date/10000000000LL%10000;
  const int month=(date/100000000%100-1)%12;
  const int day=date/1000000%100;
  const int hour=date/10000%100;
  const int min=date/100%100;
  const int sec=date%100;
  return (day-1+days[month]+(year%4==0 && month>1)+((year-1970)*1461+1)/4)
    *86400+hour*3600+min*60+sec;
}

/////////////////////////////// File //////////////////////////////////

// Windows/Linux compatible file type
#ifdef unix
typedef FILE* FP;
const FP FPNULL=NULL;
const char* const RB="rb";
const char* const WB="wb";
const char* const RBPLUS="rb+";
const char* const WBPLUS="wb+";

#else // Windows
typedef HANDLE FP;
const FP FPNULL=INVALID_HANDLE_VALUE;
typedef enum {RB, WB, RBPLUS, WBPLUS} MODE;  // fopen modes

// Open file. Only modes "rb", "wb", "rb+" and "wb+" are supported.
FP fopen(const char* filename, MODE mode) {
  assert(filename);
  DWORD access=0;
  if (mode!=WB) access=GENERIC_READ;
  if (mode!=RB) access|=GENERIC_WRITE;
  DWORD disp=OPEN_ALWAYS;  // wb or wb+
  if (mode==RB || mode==RBPLUS) disp=OPEN_EXISTING;
  DWORD share=FILE_SHARE_READ;
  if (mode==RB) share|=FILE_SHARE_WRITE|FILE_SHARE_DELETE;
  return CreateFile(utow(filename).c_str(), access, share,
                    NULL, disp, FILE_ATTRIBUTE_NORMAL, NULL);
}

// Close file
int fclose(FP fp) {
  return CloseHandle(fp) ? 0 : EOF;
}

// Read nobj objects of size size into ptr. Return number of objects read.
size_t fread(void* ptr, size_t size, size_t nobj, FP fp) {
  DWORD r=0;
  ReadFile(fp, ptr, size*nobj, &r, NULL);
  if (size>1) r/=size;
  return r;
}

// Write nobj objects of size size from ptr to fp. Return number written.
size_t fwrite(const void* ptr, size_t size, size_t nobj, FP fp) {
  DWORD r=0;
  WriteFile(fp, ptr, size*nobj, &r, NULL);
  if (size>1) r/=size;
  return r;
}

// Move file pointer by offset. origin is SEEK_SET (from start), SEEK_CUR,
// (from current position), or SEEK_END (from end).
int fseeko(FP fp, int64_t offset, int origin) {
  if (origin==SEEK_SET) origin=FILE_BEGIN;
  else if (origin==SEEK_CUR) origin=FILE_CURRENT;
  else if (origin==SEEK_END) origin=FILE_END;
  LONG h=uint64_t(offset)>>32;
  SetFilePointer(fp, offset&0xffffffffull, &h, origin);
  return GetLastError()!=NO_ERROR;
}

// Get file position
int64_t ftello(FP fp) {
  LONG h=0;
  DWORD r=SetFilePointer(fp, 0, &h, FILE_CURRENT);
  return r+(uint64_t(h)<<32);
}

#endif

// Return true if a file or directory (UTF-8 without trailing /) exists.
bool exists(string filename) {
  int len=filename.size();
  if (len<1) return false;
  if (filename[len-1]=='/') filename=filename.substr(0, len-1);
#ifdef unix
  struct stat sb;
  return !lstat(filename.c_str(), &sb);
#else
  return GetFileAttributes(utow(filename.c_str()).c_str())
         !=INVALID_FILE_ATTRIBUTES;
#endif
}

// Delete a file, return true if successful
bool delete_file(const char* filename) {
#ifdef unix
  return remove(filename)==0;
#else
  return DeleteFile(utow(filename).c_str());
#endif
}

#ifdef unix

// Print last error message
void printerr(const char* filename) {
  perror(filename);
}

#else

// Print last error message
void printerr(const char* filename) {
  fflush(stdout);
  int err=GetLastError();
  printUTF8(filename, stderr);
  if (err==ERROR_FILE_NOT_FOUND)
    fprintf(stderr, ": file not found\n");
  else if (err==ERROR_PATH_NOT_FOUND)
    fprintf(stderr, ": path not found\n");
  else if (err==ERROR_ACCESS_DENIED)
    fprintf(stderr, ": access denied\n");
  else if (err==ERROR_SHARING_VIOLATION)
    fprintf(stderr, ": sharing violation\n");
  else if (err==ERROR_BAD_PATHNAME)
    fprintf(stderr, ": bad pathname\n");
  else if (err==ERROR_INVALID_NAME)
    fprintf(stderr, ": invalid name\n");
  else if (err==ERROR_NETNAME_DELETED)
    fprintf(stderr, ": network name no longer available\n");
  else
    fprintf(stderr, ": Windows error %d\n", err);
}

#endif

// Close fp if open. Set date and attributes unless 0
void close(const char* filename, int64_t date, int64_t attr, FP fp=FPNULL) {
  assert(filename);
#ifdef unix
  if (fp!=FPNULL) fclose(fp);
  if (date>0) {
    struct utimbuf ub;
    ub.actime=time(NULL);
    ub.modtime=unix_time(date);
    utime(filename, &ub);
  }
  if ((attr&255)=='u')
    chmod(filename, attr>>8);
#else
  const bool ads=strstr(filename, ":$DATA")!=0;  // alternate data stream?
  if (date>0 && !ads) {
    if (fp==FPNULL)
      fp=CreateFile(utow(filename).c_str(),
                    FILE_WRITE_ATTRIBUTES,
                    FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (fp!=FPNULL) {
      SYSTEMTIME st;
      st.wYear=date/10000000000LL%10000;
      st.wMonth=date/100000000%100;
      st.wDayOfWeek=0;  // ignored
      st.wDay=date/1000000%100;
      st.wHour=date/10000%100;
      st.wMinute=date/100%100;
      st.wSecond=date%100;
      st.wMilliseconds=0;
      FILETIME ft;
      SystemTimeToFileTime(&st, &ft);
      SetFileTime(fp, NULL, NULL, &ft);
    }
  }
  if (fp!=FPNULL) CloseHandle(fp);
  if ((attr&255)=='w' && !ads)
    SetFileAttributes(utow(filename).c_str(), attr>>8);
#endif
}

// Print file open error and throw exception
void ioerr(const char* msg) {
  printerr(msg);
  throw std::runtime_error(msg);
}

// Create directories as needed. For example if path="/tmp/foo/bar"
// then create directories /, /tmp, and /tmp/foo unless they exist.
// Set date and attributes if not 0.
void makepath(string path, int64_t date=0, int64_t attr=0) {
  for (unsigned i=0; i<path.size(); ++i) {
    if (path[i]=='\\' || path[i]=='/') {
      path[i]=0;
#ifdef unix
      mkdir(path.c_str(), 0777);
#else
      CreateDirectory(utow(path.c_str()).c_str(), 0);
#endif
      path[i]='/';
    }
  }

  // Set date and attributes
  string filename=path;
  if (filename!="" && filename[filename.size()-1]=='/')
    filename=filename.substr(0, filename.size()-1);  // remove trailing slash
  close(filename.c_str(), date, attr);
}

#ifndef unix

// Truncate filename to length. Return -1 if error, else 0.
int truncate(const char* filename, int64_t length) {
  std::wstring w=utow(filename);
  HANDLE out=CreateFile(w.c_str(), GENERIC_READ | GENERIC_WRITE,
                        0, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (out!=INVALID_HANDLE_VALUE) {
    LONG hi=length>>32;
    if (SetFilePointer(out, length, &hi, FILE_BEGIN)
             !=INVALID_SET_FILE_POINTER
        && SetEndOfFile(out)
        && CloseHandle(out))
      return 0;
  }
  return -1;
}
#endif

/////////////////////////////// Archive ///////////////////////////////

// Convert non-negative decimal number x to string of at least n digits
string itos(int64_t x, int n=1) {
  assert(x>=0);
  assert(n>=0);
  string r;
  for (; x || n>0; x/=10, --n) r=string(1, '0'+x%10)+r;
  return r;
}

// Replace * and ? in fn with part or digits of part
string subpart(string fn, int part) {
  for (int j=fn.size()-1; j>=0; --j) {
    if (fn[j]=='?')
      fn[j]='0'+part%10, part/=10;
    else if (fn[j]=='*')
      fn=fn.substr(0, j)+itos(part)+fn.substr(j+1), part=0;
  }
  return fn;
}

// Base of InputArchive and OutputArchive
class ArchiveBase {
protected:
  libzpaq::AES_CTR* aes;  // NULL if not encrypted
  FP fp;          // currently open file or FPNULL
public:
  ArchiveBase(): aes(0), fp(FPNULL) {}
  ~ArchiveBase() {
    if (aes) delete aes;
    if (fp!=FPNULL) fclose(fp);
  }  
  bool isopen() {return fp!=FPNULL;}
};

// An InputArchive supports encrypted reading
class InputArchive: public ArchiveBase, public libzpaq::Reader {
  vector<int64_t> sz;  // part sizes
  int64_t off;  // current offset
  string fn;  // filename, possibly multi-part with wildcards
public:

  // Open filename. If password then decrypt input.
  InputArchive(const char* filename, const char* password=0);

  // Read and return 1 byte or -1 (EOF)
  int get() {
    error("get() not implemented");
    return -1;
  }

  // Read up to len bytes into obuf at current offset. Return 0..len bytes
  // actually read. 0 indicates EOF.
  int read(char* obuf, int len) {
    int nr=fread(obuf, 1, len, fp);
    if (nr==0) {
      seek(0, SEEK_CUR);
      nr=fread(obuf, 1, len, fp);
    }
    if (nr==0) return 0;
    if (aes) aes->encrypt(obuf, nr, off);
    off+=nr;
    return nr;
  }

  // Like fseeko()
  void seek(int64_t p, int whence);

  // Like ftello()
  int64_t tell() {
    return off;
  }
};

// Like fseeko. If p is out of range then close file.
void InputArchive::seek(int64_t p, int whence) {
  if (!isopen()) return;

  // Compute new offset
  if (whence==SEEK_SET) off=p;
  else if (whence==SEEK_CUR) off+=p;
  else if (whence==SEEK_END) {
    off=p;
    for (unsigned i=0; i<sz.size(); ++i) off+=sz[i];
  }

  // Optimization for single file to avoid close and reopen
  if (sz.size()==1) {
    fseeko(fp, off, SEEK_SET);
    return;
  }

  // Seek across multiple files
  assert(sz.size()>1);
  int64_t sum=0;
  unsigned i;
  for (i=0;; ++i) {
    sum+=sz[i];
    if (sum>off || i+1>=sz.size()) break;
  }
  const string next=subpart(fn, i+1);
  fclose(fp);
  fp=fopen(next.c_str(), RB);
  if (fp==FPNULL) ioerr(next.c_str());
  fseeko(fp, off-sum, SEEK_END);
}

// Open for input. Decrypt with password and using the salt in the
// first 32 bytes. If filename has wildcards then assume multi-part
// and read their concatenation.

InputArchive::InputArchive(const char* filename, const char* password):
    off(0), fn(filename) {
  assert(filename);

  // Get file sizes
  const string part0=subpart(filename, 0);
  for (unsigned i=1; ; ++i) {
    const string parti=subpart(filename, i);
    if (i>1 && parti==part0) break;
    fp=fopen(parti.c_str(), RB);
    if (fp==FPNULL) break;
    fseeko(fp, 0, SEEK_END);
    sz.push_back(ftello(fp));
    fclose(fp);
  }

  // Open first part
  const string part1=subpart(filename, 1);
  fp=fopen(part1.c_str(), RB);
  if (!isopen()) ioerr(part1.c_str());
  assert(fp!=FPNULL);

  // Get encryption salt
  if (password) {
    char salt[32], key[32];
    if (fread(salt, 1, 32, fp)!=32) error("cannot read salt");
    libzpaq::stretchKey(key, password, salt);
    aes=new libzpaq::AES_CTR(key, 32, salt);
    off=32;
  }
}

// An Archive is a file supporting encryption
class OutputArchive: public ArchiveBase, public libzpaq::Writer {
  int64_t off;    // preceding multi-part bytes
  unsigned ptr;   // write pointer in buf: 0 <= ptr <= BUFSIZE
  enum {BUFSIZE=1<<16};
  char buf[BUFSIZE];  // I/O buffer
public:

  // Open. If password then encrypt output.
  OutputArchive(const char* filename, const char* password=0,
                const char* salt_=0, int64_t off_=0);

  // Write pending output
  void flush() {
    assert(fp!=FPNULL);
    if (aes) aes->encrypt(buf, ptr, ftello(fp)+off);
    fwrite(buf, 1, ptr, fp);
    ptr=0;
  }

  // Position the next read or write offset to p.
  void seek(int64_t p, int whence) {
    if (fp!=FPNULL) {
      flush();
      fseeko(fp, p, whence);
    }
    else if (whence==SEEK_SET) off=p;
    else off+=p;  // assume at end
  }

  // Return current file offset.
  int64_t tell() const {
    if (fp!=FPNULL) return ftello(fp)+ptr;
    else return off;
  }

  // Write one byte
  void put(int c) {
    if (fp==FPNULL) ++off;
    else {
      if (ptr>=BUFSIZE) flush();
      buf[ptr++]=c;
    }
  }

  // Write buf[0..n-1]
  void write(const char* ibuf, int len) {
    if (fp==FPNULL) off+=len;
    else while (len-->0) put(*ibuf++);
  }

  // Flush output and close
  void close() {
    if (fp!=FPNULL) {
      flush();
      fclose(fp);
    }
    fp=FPNULL;
  }
};

// Create or update an existing archive or part. If filename is ""
// then keep track of position in off but do not write to disk. Otherwise
// open and encrypt with password if not 0. If the file exists then
// read the salt from the first 32 bytes and off_ must be 0. Otherwise
// encrypt assuming off_ previous bytes, of which the first 32 are salt_.
// If off_ is 0 then write salt_ to the first 32 bytes.

OutputArchive::OutputArchive(const char* filename, const char* password,
    const char* salt_, int64_t off_): off(off_), ptr(0) {
  assert(filename);
  if (!*filename) return;

  // Open existing file
  char salt[32]={0};
  fp=fopen(filename, RBPLUS);
  if (isopen()) {
    if (off!=0) error("file exists and off > 0");
    if (password) {
      if (fread(salt, 1, 32, fp)!=32) error("cannot read salt");
      if (salt_ && memcmp(salt, salt_, 32)) error("salt mismatch");
    }
    seek(0, SEEK_END);
  }

  // Create new file
  else {
    fp=fopen(filename, WB);
    if (!isopen()) ioerr(filename);
    if (password) {
      if (!salt_) error("salt not specified");
      memcpy(salt, salt_, 32);
      if (off==0 && fwrite(salt, 1, 32, fp)!=32) ioerr(filename);
    }
  }

  // Set up encryption
  if (password) {
    char key[32];
    libzpaq::stretchKey(key, password, salt);
    aes=new libzpaq::AES_CTR(key, 32, salt);
  }
}

///////////////////////// System info /////////////////////////////////

// Guess number of cores. In 32 bit mode, max is 2.
int numberOfProcessors() {
  int rc=0;  // result
#ifdef unix
#ifdef BSD  // BSD or Mac OS/X
  size_t rclen=sizeof(rc);
  int mib[2]={CTL_HW, HW_NCPU};
  if (sysctl(mib, 2, &rc, &rclen, 0, 0)!=0)
    perror("sysctl");

#else  // Linux
  // Count lines of the form "processor\t: %d\n" in /proc/cpuinfo
  // where %d is 0, 1, 2,..., rc-1
  FILE *in=fopen("/proc/cpuinfo", "r");
  if (!in) return 1;
  std::string s;
  int c;
  while ((c=getc(in))!=EOF) {
    if (c>='A' && c<='Z') c+='a'-'A';  // convert to lowercase
    if (c>' ') s+=c;  // remove white space
    if (c=='\n') {  // end of line?
      if (s.size()>10 && s.substr(0, 10)=="processor:") {
        c=atoi(s.c_str()+10);
        if (c==rc) ++rc;
      }
      s="";
    }
  }
  fclose(in);
#endif
#else

  // In Windows return %NUMBER_OF_PROCESSORS%
  const char* p=getenv("NUMBER_OF_PROCESSORS");
  if (p) rc=atoi(p);
#endif
  if (rc<1) rc=1;
  if (sizeof(char*)==4 && rc>2) rc=2;
  return rc;
}

////////////////////////////// misc ///////////////////////////////////

// For libzpaq output to a string less than 64K chars
struct StringWriter: public libzpaq::Writer {
  string s;
  void put(int c) {
    if (s.size()>=65535) error("string too long");
    s+=char(c);
  }
};

// In Windows convert upper case to lower case.
inline int tolowerW(int c) {
#ifndef unix
  if (c>='A' && c<='Z') return c-'A'+'a';
#endif
  return c;
}

// Return true if strings a == b or a+"/" is a prefix of b
// or a ends in "/" and is a prefix of b.
// Match ? in a to any char in b.
// Match * in a to any string in b.
// In Windows, not case sensitive.
bool ispath(const char* a, const char* b) {
  for (; *a; ++a, ++b) {
    const int ca=tolowerW(*a);
    const int cb=tolowerW(*b);
    if (ca=='*') {
      while (true) {
        if (ispath(a+1, b)) return true;
        if (!*b) return false;
        ++b;
      }
    }
    else if (ca=='?') {
      if (*b==0) return false;
    }
    else if (ca==cb && ca=='/' && a[1]==0)
      return true;
    else if (ca!=cb)
      return false;
  }
  return *b==0 || *b=='/';
}

// Read 4 byte little-endian int and advance s
unsigned btoi(const char* &s) {
  s+=4;
  return (s[-4]&255)|((s[-3]&255)<<8)|((s[-2]&255)<<16)|((s[-1]&255)<<24);
}

// Read 8 byte little-endian int and advance s
int64_t btol(const char* &s) {
  uint64_t r=btoi(s);
  return r+(uint64_t(btoi(s))<<32);
}

/////////////////////////////// Jidac /////////////////////////////////

// A Jidac object represents an archive contents: a list of file
// fragments with hash, size, and archive offset, and a list of
// files with date, attributes, and list of fragment pointers.
// Methods add to, extract from, compare, and list the archive.

// enum for version
static const int64_t DEFAULT_VERSION=99999999999999LL; // unless -until

// fragment hash table entry
struct HT {
  unsigned char sha1[20];  // fragment hash
  int usize;      // uncompressed size, -1 if unknown, -2 if not init
  HT(const char* s=0, int u=-2) {
    if (s) memcpy(sha1, s, 20);
    else memset(sha1, 0, 20);
    usize=u;
  }
};

// filename entry
struct DT {
  int64_t date;          // decimal YYYYMMDDHHMMSS (UT) or 0 if deleted
  int64_t size;          // size or -1 if unknown
  int64_t attr;          // first 8 attribute bytes
  int64_t data;          // sort key or frags written. -1 = do not write
  vector<unsigned> ptr;  // fragment list
  DT(): date(0), size(0), attr(0), data(0) {}
};
typedef map<string, DT> DTMap;

// list of blocks to extract
struct Block {
  int64_t offset;       // location in archive
  int64_t usize;        // uncompressed size, -1 if unknown (streaming)
  int64_t bsize;        // compressed size
  vector<DTMap::iterator> files;  // list of files pointing here
  unsigned start;       // index in ht of first fragment
  unsigned size;        // number of fragments to decompress
  unsigned frags;       // number of fragments in block
  unsigned extracted;   // number of fragments decompressed OK
  enum {READY, WORKING, GOOD, BAD} state;
  Block(unsigned s, int64_t o): offset(o), usize(-1), bsize(0), start(s),
      size(0), frags(0), extracted(0), state(READY) {}
};

// Version info
struct VER {
  int64_t date;          // Date of C block, 0 if streaming
  int64_t lastdate;      // Latest date of any block
  int64_t offset;        // start of transaction C block
  int64_t data_offset;   // start of first D block
  int64_t csize;         // size of compressed data, -1 = no index
  int updates;           // file updates
  int deletes;           // file deletions
  unsigned firstFragment;// first fragment ID
  VER() {memset(this, 0, sizeof(*this));}
};

// Windows API functions not in Windows XP to be dynamically loaded
#ifndef unix
typedef HANDLE (WINAPI* FindFirstStreamW_t)
                   (LPCWSTR, STREAM_INFO_LEVELS, LPVOID, DWORD);
FindFirstStreamW_t findFirstStreamW=0;
typedef BOOL (WINAPI* FindNextStreamW_t)(HANDLE, LPVOID);
FindNextStreamW_t findNextStreamW=0;
#endif

class CompressJob;

// Do everything
class Jidac {
public:
  int doCommand(int argc, const char** argv);
  friend ThreadReturn decompressThread(void* arg);
  friend ThreadReturn testThread(void* arg);
  friend struct ExtractJob;
private:

  // Command line arguments
  char command;             // command 'a', 'x', or 'l'
  string archive;           // archive name
  vector<string> files;     // filename args
  int all;                  // -all option
  bool force;               // -force option
  int fragment;             // -fragment option
  const char* index;        // index option
  char password_string[32]; // hash of -key argument
  const char* password;     // points to password_string or NULL
  string method;            // default "1"
  bool noattributes;        // -noattributes option
  vector<string> notfiles;  // list of prefixes to exclude
  string nottype;           // -not =...
  vector<string> onlyfiles; // list of prefixes to include
  const char* repack;       // -repack output file
  char new_password_string[32]; // -repack hashed password
  const char* new_password; // points to new_password_string or NULL
  int summary;              // summary option if > 0, detailed if -1
  bool dotest;              // -test option
  int threads;              // default is number of cores
  vector<string> tofiles;   // -to option
  int64_t date;             // now as decimal YYYYMMDDHHMMSS (UT)
  int64_t version;          // version number or 14 digit date

  // Archive state
  int64_t dhsize;           // total size of D blocks according to H blocks
  int64_t dcsize;           // total size of D blocks according to C blocks
  vector<HT> ht;            // list of fragments
  DTMap dt;                 // set of files in archive
  DTMap edt;                // set of external files to add or compare
  vector<Block> block;      // list of data blocks to extract
  vector<VER> ver;          // version info

  // Commands
  int add();                // add, return 1 if error else 0
  int extract();            // extract, return 1 if error else 0
  int list();               // list, return 0
  void usage();             // help

  // Support functions
  string rename(string name);           // rename from -to
  int64_t read_archive(const char* arc, int *errors=0);  // read arc
  bool isselected(const char* filename, bool rn=false);// files, -only, -not
  void scandir(string filename);        // scan dirs to dt
  void addfile(string filename, int64_t edate, int64_t esize,
               int64_t eattr);          // add external file to dt
  void list_versions(int64_t csize);    // print ver. csize=archive size
  bool equal(DTMap::const_iterator p, const char* filename);
             // compare file contents with p
};

// Print help message
void Jidac::usage() {
  printf(
"Usage: zpaq command archive[.zpaq] files... -options...\n"
"Files... may be directory trees. Default is the whole archive.\n"
"Use * or \?\?\?\? in archive name for multi-part or \"\" for empty.\n"
"Commands:\n"
"   a  add         Append files to archive if dates have changed.\n"
"   x  extract     Extract most recent versions of files.\n"
"   l  list        List or compare external files to archive by dates.\n"
"Options:\n"
"  -all [N]        Extract/list versions in N [4] digit directories.\n"
"  -f -force       Add: append files if contents have changed.\n"
"                  Extract: overwrite existing output files.\n"
"                  List: compare file contents instead of dates.\n"
"  -index F        Extract: create index F for archive.\n"
"                  Add: create suffix for archive indexed by F, update F.\n"
"  -key X          Create or access encrypted archive with password X.\n"
"  -mN  -method N  Compress level N (0..5 = faster..better, default 1).\n"
"  -noattributes   Ignore/don't save file attributes or permissions.\n"
"  -not files...   Exclude. * and ? match any string or char.\n"
"       =[+-#^?]   List: exclude by comparison result.\n"
"  -only files...  Include only matches (default: *).\n"
"  -repack F [X]   Extract to new archive F with key X (default: none).\n"
"  -sN -summary N  List: show top N sorted by size. -1: show frag IDs.\n"
"                  Add/Extract: if N > 0 show brief progress.\n"
"  -test           Extract: verify but do not write files.\n"
"  -tN -threads N  Use N threads (default: 0 = %d cores).\n"
"  -to out...      Rename files... to out... or all to out/all.\n"
"  -until N        Roll back archive to N'th update or -N from end.\n"
"  -until %s  Set date, roll back (UT, default time: 235959).\n"
#ifndef NDEBUG
"Advanced options:\n"
"  -fragment N     Use 2^N KiB average fragment size (default: 6).\n"
"  -mNB -method NB Use 2^B MiB blocks (0..11, default: 04, 14, 26..56).\n"
"  -method {xs}B[,N2]...[{ciawmst}[N1[,N2]...]]...  Advanced:\n"
"  x=journaling (default). s=streaming (no dedupe).\n"
"    N2: 0=no pre/post. 1,2=packed,byte LZ77. 3=BWT. 4..7=0..3 with E8E9.\n"
"    N3=LZ77 min match. N4=longer match to try first (0=none). 2^N5=search\n"
"    depth. 2^N6=hash table size (N6=B+21: suffix array). N7=lookahead.\n"
"    Context modeling defaults shown below:\n"
"  c0,0,0: context model. N1: 0=ICM, 1..256=CM max count. 1000..1256 halves\n"
"    memory. N2: 1..255=offset mod N2, 1000..1255=offset from N2-1000 byte.\n"
"    N3...: order 0... context masks (0..255). 256..511=mask+byte LZ77\n"
"    parse state, >1000: gap of N3-1000 zeros.\n"
"  i: ISSE chain. N1=context order. N2...=order increment.\n"
"  a24,0,0: MATCH: N1=hash multiplier. N2=halve buffer. N3=halve hash tab.\n"
"  w1,65,26,223,20,0: Order 0..N1-1 word ISSE chain. A word is bytes\n"
"    N2..N2+N3-1 ANDed with N4, hash mulitpiler N5, memory halved by N6.\n"
"  m8,24: MIX all previous models, N1 context bits, learning rate N2.\n"
"  s8,32,255: SSE last model. N1 context bits, count range N2..N3.\n"
"  t8,24: MIX2 last 2 models, N1 context bits, learning rate N2.\n"
#endif
  , threads, dateToString(date).c_str());
  exit(1);
}

// return a/b such that there is exactly one "/" in between, and
// in Windows, any drive letter in b the : is removed and there
// is a "/" after.
string append_path(string a, string b) {
  int na=a.size();
  int nb=b.size();
#ifndef unix
  if (nb>1 && b[1]==':') {  // remove : from drive letter
    if (nb>2 && b[2]!='/') b[1]='/';
    else b=b[0]+b.substr(2), --nb;
  }
#endif
  if (nb>0 && b[0]=='/') b=b.substr(1);
  if (na>0 && a[na-1]=='/') a=a.substr(0, na-1);
  return a+"/"+b;
}

// Rename name using tofiles[]
string Jidac::rename(string name) {
  if (files.size()==0 && tofiles.size()>0)  // append prefix tofiles[0]
    name=append_path(tofiles[0], name);
  else {  // replace prefix files[i] with tofiles[i]
    const int n=name.size();
    for (unsigned i=0; i<files.size() && i<tofiles.size(); ++i) {
      const int fn=files[i].size();
      if (fn<=n && files[i]==name.substr(0, fn))
        return tofiles[i]+name.substr(fn);
    }
  }
  return name;
}

// Parse the command line. Return 1 if error else 0.
int Jidac::doCommand(int argc, const char** argv) {

  // Initialize options to default values
  command=0;
  force=false;
  fragment=6;
  all=0;
  password=0;  // no password
  index=0;
  method="";  // 0..5
  noattributes=false;
  repack=0;
  new_password=0;
  summary=0; // detailed: -1
  dotest=false;  // -test
  threads=0; // 0 = auto-detect
  version=DEFAULT_VERSION;
  date=0;

  printf("zpaq v" ZPAQ_VERSION " journaling archiver, compiled "
         __DATE__ "\n");

  // Init archive state
  ht.resize(1);  // element 0 not used
  ver.resize(1); // version 0
  dhsize=dcsize=0;

  // Get date
  time_t now=time(NULL);
  tm* t=gmtime(&now);
  date=(t->tm_year+1900)*10000000000LL+(t->tm_mon+1)*100000000LL
      +t->tm_mday*1000000+t->tm_hour*10000+t->tm_min*100+t->tm_sec;

  // Get optional options
  for (int i=1; i<argc; ++i) {
    const string opt=argv[i];  // read command
    if ((opt=="add" || opt=="extract" || opt=="list" || opt=="convert"
         || opt=="a" || opt=="x" || opt=="l" || opt=="c")
        && i<argc-1 && argv[i+1][0]!='-' && command==0) {
      command=opt[0];
      if (opt=="extract") command='x';
      archive=argv[++i];  // append ".zpaq" to archive if no extension
      const char* slash=strrchr(argv[i], '/');
      const char* dot=strrchr(slash ? slash : argv[i], '.');
      if (!dot && archive!="") archive+=".zpaq";
      while (++i<argc && argv[i][0]!='-')  // read filename args
        files.push_back(argv[i]);
      --i;
    }
    else if (opt.size()<2 || opt[0]!='-') usage();
    else if (opt=="-all") {
      all=4;
      if (i<argc-1 && isdigit(argv[i+1][0])) all=atoi(argv[++i]);
    }
    else if (opt=="-force" || opt=="-f") force=true;
    else if (opt=="-fragment" && i<argc-1) fragment=atoi(argv[++i]);
    else if (opt=="-index" && i<argc-1) index=argv[++i];
    else if (opt=="-key" && i<argc-1) {
      libzpaq::SHA256 sha256;
      for (const char* p=argv[++i]; *p; ++p) sha256.put(*p);
      memcpy(password_string, sha256.result(), 32);
      password=password_string;
    }
    else if (opt=="-method" && i<argc-1) method=argv[++i];
    else if (opt[1]=='m') method=argv[i]+2;
    else if (opt=="-noattributes") noattributes=true;
    else if (opt=="-not") {  // read notfiles
      while (++i<argc && argv[i][0]!='-') {
        if (argv[i][0]=='=') nottype=argv[i];
        else notfiles.push_back(argv[i]);
      }
      --i;
    }
    else if (opt=="-only") {  // read onlyfiles
      while (++i<argc && argv[i][0]!='-')
        onlyfiles.push_back(argv[i]);
      --i;
    }
    else if (opt=="-repack" && i<argc-1) {
      repack=argv[++i];
      if (i<argc-1 && argv[i+1][0]!='-') {
        libzpaq::SHA256 sha256;
        for (const char* p=argv[++i]; *p; ++p) sha256.put(*p);
        memcpy(new_password_string, sha256.result(), 32);
        new_password=new_password_string;
      }
    }
    else if (opt=="-summary" && i<argc-1) summary=atoi(argv[++i]);
    else if (opt[1]=='s') summary=atoi(argv[i]+2);
    else if (opt=="-test") dotest=true;
    else if (opt=="-to") {  // read tofiles
      while (++i<argc && argv[i][0]!='-')
        tofiles.push_back(argv[i]);
      if (tofiles.size()==0) tofiles.push_back("");
      --i;
    }
    else if (opt=="-threads" && i<argc-1) threads=atoi(argv[++i]);
    else if (opt[1]=='t') threads=atoi(argv[i]+2);
    else if (opt=="-until" && i+1<argc) {  // read date

      // Read digits from multiple args and fill in leading zeros
      version=0;
      int digits=0;
      if (argv[i+1][0]=='-') {  // negative version
        version=atol(argv[i+1]);
        if (version>-1) usage();
        ++i;
      }
      else {  // positive version or date
        while (++i<argc && argv[i][0]!='-') {
          for (int j=0; ; ++j) {
            if (isdigit(argv[i][j])) {
              version=version*10+argv[i][j]-'0';
              ++digits;
            }
            else {
              if (digits==1) version=version/10*100+version%10;
              digits=0;
              if (argv[i][j]==0) break;
            }
          }
        }
        --i;
      }

      // Append default time
      if (version>=19000000LL     && version<=29991231LL)
        version=version*100+23;
      if (version>=1900000000LL   && version<=2999123123LL)
        version=version*100+59;
      if (version>=190000000000LL && version<=299912312359LL)
        version=version*100+59;
      if (version>9999999) {
        if (version<19000101000000LL || version>29991231235959LL) {
          fflush(stdout);
          fprintf(stderr,
            "Version date %1.0f must be 19000101000000 to 29991231235959\n",
             double(version));
          exit(1);
        }
        date=version;
      }
    }
    else {
      printf("Unknown option ignored: %s\n", argv[i]);
      usage();
    }
  }

  // Set threads
  if (threads<1) threads=numberOfProcessors();

  // Test date
  if (now==-1 || date<19000000000000LL || date>30000000000000LL)
    error("date is incorrect, use -until YYYY-MM-DD HH:MM:SS to set");

  // Adjust negative version
  if (version<0) {
    Jidac jidac(*this);
    jidac.version=DEFAULT_VERSION;
    jidac.read_archive(archive.c_str());
    version+=jidac.ver.size()-1;
    printf("Version %1.0f\n", version+.0);
  }

  // Load dynamic functions in Windows Vista and later
#ifndef unix
  HMODULE h=GetModuleHandle(TEXT("kernel32.dll"));
  if (h==NULL) printerr("GetModuleHandle");
  else {
    findFirstStreamW=
      (FindFirstStreamW_t)GetProcAddress(h, "FindFirstStreamW");
    findNextStreamW=
      (FindNextStreamW_t)GetProcAddress(h, "FindNextStreamW");
  }
  if (!findFirstStreamW || !findNextStreamW)
    printf("Alternate streams not supported in Windows XP.\n");
#endif

  // Execute command
  if (command=='a' && files.size()>0) return add();
  else if (command=='x') return extract();
  else if (command=='l') list();
  else usage();
  return 0;
}

/////////////////////////// read_archive //////////////////////////////

// Read arc up to -date into ht, dt, ver. Return place to
// append. If errors is not NULL then set it to number of errors found.
int64_t Jidac::read_archive(const char* arc, int *errors) {
  if (errors) *errors=0;
  dcsize=dhsize=0;
  assert(ver.size()==1);
  unsigned files=0;  // count

  // Open archive
  InputArchive in(arc, password);
  if (!in.isopen()) {
    if (command!='a') {
      fflush(stdout);
      printUTF8(arc, stderr);
      fprintf(stderr, " not found.\n");
      if (errors) ++*errors;
    }
    return 0;
  }
  printUTF8(arc);
  if (version==DEFAULT_VERSION) printf(": ");
  else printf(" -until %1.0f: ", version+0.0);
  fflush(stdout);

  // Test password
  {
    char s[4]={0};
    const int nr=in.read(s, 4);
    if (nr>0 && memcmp(s, "7kSt", 4) && (memcmp(s, "zPQ", 3) || s[3]<1))
      error("password incorrect");
    in.seek(-nr, SEEK_CUR);
  }

  // Scan archive contents
  string lastfile=archive; // last named file in streaming format
  if (lastfile.size()>5 && lastfile.substr(lastfile.size()-5)==".zpaq")
    lastfile=lastfile.substr(0, lastfile.size()-5); // drop .zpaq
  int64_t block_offset=32*(password!=0);  // start of last block of any type
  int64_t data_offset=block_offset;    // start of last block of d fragments
  bool found_data=false;   // exit if nothing found
  bool first=true;         // first segment in archive?
  StringBuffer os(32832);  // decompressed block
  const bool renamed=command=='l' || command=='a';

  // Detect archive format and read the filenames, fragment sizes,
  // and hashes. In JIDAC format, these are in the index blocks, allowing
  // data to be skipped. Otherwise the whole archive is scanned to get
  // this information from the segment headers and trailers.
  bool done=false;
  while (!done) {
    libzpaq::Decompresser d;
    try {
      d.setInput(&in);
      double mem=0;
      while (d.findBlock(&mem)) {
        found_data=true;

        // Read the segments in the current block
        StringWriter filename, comment;
        int segs=0;  // segments in block
        bool skip=false;  // skip decompression?
        while (d.findFilename(&filename)) {
          if (filename.s.size()) {
            for (unsigned i=0; i<filename.s.size(); ++i)
              if (filename.s[i]=='\\') filename.s[i]='/';
            lastfile=filename.s.c_str();
          }
          comment.s="";
          d.readComment(&comment);

          // Test for JIDAC format. Filename is jDC<fdate>[cdhi]<num>
          // and comment ends with " jDC\x01". Skip d (data) blocks.
          if (comment.s.size()>=4
              && comment.s.substr(comment.s.size()-4)=="jDC\x01") {
            if (filename.s.size()!=28 || filename.s.substr(0, 3)!="jDC")
              error("bad journaling block name");
            if (skip) error("mixed journaling and streaming block");

            // Read uncompressed size from comment
            int64_t usize=0;
            unsigned i;
            for (i=0; i<comment.s.size() && isdigit(comment.s[i]); ++i) {
              usize=usize*10+comment.s[i]-'0';
              if (usize>0xffffffff) error("journaling block too big");
            }

            // Read the date and number in the filename
            int64_t fdate=0, num=0;
            for (i=3; i<17 && isdigit(filename.s[i]); ++i)
              fdate=fdate*10+filename.s[i]-'0';
            if (i!=17 || fdate<19000000000000LL || fdate>=30000000000000LL)
              error("bad date");
            for (i=18; i<28 && isdigit(filename.s[i]); ++i)
              num=num*10+filename.s[i]-'0';
            if (i!=28 || num>0xffffffff) error("bad fragment");

            // Decompress the block.
            os.resize(0);
            os.setLimit(usize);
            d.setOutput(&os);
            libzpaq::SHA1 sha1;
            d.setSHA1(&sha1);
            if (strchr("chi", filename.s[17])) {
              if (mem>1.5e9) error("index block requires too much memory");
              d.decompress();
              char sha1result[21]={0};
              d.readSegmentEnd(sha1result);
              if ((int64_t)os.size()!=usize) error("bad block size");
              if (usize!=int64_t(sha1.usize())) error("bad checksum size");
              if (sha1result[0] && memcmp(sha1result+1, sha1.result(), 20))
                error("bad checksum");
            }
            else
              d.readSegmentEnd();

            // Transaction header (type c).
            // If in the future then stop here, else read 8 byte data size
            // from input and jump over it.
            if (filename.s[17]=='c') {
              if (os.size()<8) error("c block too small");
              data_offset=in.tell()+1-d.buffered();
              const char* s=os.c_str();
              int64_t jmp=btol(s);
              if (jmp<0) printf("Incomplete transaction ignored\n");
              if (jmp<0
                  || (version<19000000000000LL && int64_t(ver.size())>version)
                  || (version>=19000000000000LL && version<fdate)) {
                done=true;  // roll back to here
                goto endblock;
              }
              else {
                dcsize+=jmp;
                if (jmp) in.seek(data_offset+jmp, SEEK_SET);
                ver.push_back(VER());
                ver.back().firstFragment=ht.size();
                ver.back().offset=block_offset;
                ver.back().data_offset=data_offset;
                ver.back().date=ver.back().lastdate=fdate;
                ver.back().csize=jmp;
                if (all) {
                  string fn=itos(ver.size()-1, all)+"/";
                  if (renamed) fn=rename(fn);
                  if (isselected(fn.c_str(), false))
                    dt[fn].date=fdate;
                }
                if (jmp) goto endblock;
              }
            }

            // Fragment table (type h).
            // Contents is bsize[4] (sha1[20] usize[4])... for fragment N...
            // where bsize is the compressed block size.
            // Store in ht[].{sha1,usize}. Set ht[].csize to block offset
            // assuming N in ascending order.
            else if (filename.s[17]=='h') {
              assert(ver.size()>0);
              if (fdate>ver.back().lastdate) ver.back().lastdate=fdate;
              if (os.size()%24!=4) error("bad h block size");
              const unsigned n=(os.size()-4)/24;
              if (num<1 || num+n>0xffffffff) error("bad h fragment");
              const char* s=os.c_str();
              const unsigned bsize=btoi(s);
              dhsize+=bsize;
              assert(ver.size()>0);
              if (int64_t(ht.size())>num) {
                fflush(stdout);
                fprintf(stderr,
                  "Unordered fragment tables: expected >= %d found %1.0f\n",
                  int(ht.size()), double(num));
              }
              for (unsigned i=0; i<n; ++i) {
                if (i==0) {
                  block.push_back(Block(num, data_offset));
                  block.back().usize=8;
                  block.back().bsize=bsize;
                  block.back().frags=os.size()/24;
                }
                while (int64_t(ht.size())<=num+i) ht.push_back(HT());
                memcpy(ht[num+i].sha1, s, 20);
                s+=20;
                assert(block.size()>0);
                unsigned f=btoi(s);
                if (f>0x7fffffff) error("fragment too big");
                block.back().usize+=(ht[num+i].usize=f)+4u;
              }
              data_offset+=bsize;
            }

            // Index (type i)
            // Contents is: 0[8] filename 0 (deletion)
            // or:       date[8] filename 0 na[4] attr[na] ni[4] ptr[ni][4]
            // Read into DT
            else if (filename.s[17]=='i') {
              assert(ver.size()>0);
              if (fdate>ver.back().lastdate) ver.back().lastdate=fdate;
              const char* s=os.c_str();
              const char* const end=s+os.size();
              while (s+9<=end) {
                DT dtr;
                dtr.date=btol(s);  // date
                if (dtr.date) ++ver.back().updates;
                else ++ver.back().deletes;
                const int64_t len=strlen(s);
                if (len>65535) error("filename too long");
                string fn=s;  // filename renamed
                if (all) fn=append_path(itos(ver.size()-1, all), fn);
                const bool issel=isselected(fn.c_str(), renamed);
                s+=len+1;  // skip filename
                if (s>end) error("filename too long");
                if (dtr.date) {
                  ++files;
                  if (s+4>end) error("missing attr");
                  unsigned na=btoi(s);  // attr bytes
                  if (s+na>end || na>65535) error("attr too long");
                  for (unsigned i=0; i<na; ++i, ++s)  // read attr
                    if (i<8) dtr.attr+=int64_t(*s&255)<<(i*8);
                  if (noattributes) dtr.attr=0;
                  if (s+4>end) error("missing ptr");
                  unsigned ni=btoi(s);  // ptr list size
                  if (ni>(end-s)/4u) error("ptr list too long");
                  if (issel) dtr.ptr.resize(ni);
                  for (unsigned i=0; i<ni; ++i) {  // read ptr
                    const unsigned j=btoi(s);
                    if (issel) dtr.ptr[i]=j;
                  }
                }
                if (issel) dt[fn]=dtr;
              }  // end while more files
            }  // end if 'i'
            else {
              printf("Skipping %s %s\n",
                  filename.s.c_str(), comment.s.c_str());
              error("Unexpected journaling block");
            }
          }  // end if journaling

          // Streaming format
          else {

            // If previous version does not exist, start a new one
            if (ver.size()==1) {
              if (version<1) {
                done=true;
                goto endblock;
              }
              ver.push_back(VER());
              ver.back().firstFragment=ht.size();
              ver.back().offset=block_offset;
              ver.back().csize=-1;
            }

            char sha1result[21]={0};
            d.readSegmentEnd(sha1result);
            skip=true;
            string fn=lastfile;
            if (all) fn=append_path(itos(ver.size()-1, all), fn);
            if (isselected(fn.c_str(), renamed)) {
              DT& dtr=dt[fn];
              if (filename.s.size()>0 || first) {
                ++files;
                dtr.date=date;
                dtr.attr=0;
                dtr.ptr.resize(0);
                ++ver.back().updates;
              }
              dtr.ptr.push_back(ht.size());
            }
            assert(ver.size()>0);
            if (segs==0 || block.size()==0)
              block.push_back(Block(ht.size(), block_offset));
            assert(block.size()>0);
            ht.push_back(HT(sha1result+1, -1));
          }  // end else streaming
          ++segs;
          filename.s="";
          first=false;
        }  // end while findFilename
        if (!done) block_offset=in.tell()-d.buffered();
      }  // end while findBlock
      done=true;
    }  // end try
    catch (std::exception& e) {
      in.seek(-d.buffered(), SEEK_CUR);
      fflush(stdout);
      fprintf(stderr, "Skipping block at %1.0f: %s\n", double(block_offset),
              e.what());
      if (errors) ++*errors;
    }
endblock:;
  }  // end while !done
  if (in.tell()>32*(password!=0) && !found_data)
    error("archive contains no data");
  printf("%d versions, %u files, %u fragments, %1.6f MB\n", 
      int(ver.size()-1), files, unsigned(ht.size())-1,
      block_offset/1000000.0);

  // Calculate file sizes
  for (DTMap::iterator p=dt.begin(); p!=dt.end(); ++p) {
    for (unsigned i=0; i<p->second.ptr.size(); ++i) {
      unsigned j=p->second.ptr[i];
      if (j>0 && j<ht.size() && p->second.size>=0) {
        if (ht[j].usize>=0) p->second.size+=ht[j].usize;
        else p->second.size=-1;  // unknown size
      }
    }
  }
  return block_offset;
}

// Test whether filename and attributes are selected by files, -only, and -not
// If rn then test renamed filename.
bool Jidac::isselected(const char* filename, bool rn) {
  bool matched=true;
  if (files.size()>0) {
    matched=false;
    for (unsigned i=0; i<files.size() && !matched; ++i) {
      if (rn && i<tofiles.size()) {
        if (ispath(tofiles[i].c_str(), filename)) matched=true;
      }
      else if (ispath(files[i].c_str(), filename)) matched=true;
    }
  }
  if (!matched) return false;
  if (onlyfiles.size()>0) {
    matched=false;
    for (unsigned i=0; i<onlyfiles.size() && !matched; ++i)
      if (ispath(onlyfiles[i].c_str(), filename))
        matched=true;
  }
  if (!matched) return false;
  for (unsigned i=0; i<notfiles.size(); ++i) {
    if (ispath(notfiles[i].c_str(), filename))
      return false;
  }
  return true;
}

// Return the part of fn up to the last slash
string path(const string& fn) {
  int n=0;
  for (int i=0; fn[i]; ++i)
    if (fn[i]=='/' || fn[i]=='\\') n=i+1;
  return fn.substr(0, n);
}

// Insert external filename (UTF-8 with "/") into dt if selected
// by files, onlyfiles, and notfiles. If filename
// is a directory then also insert its contents.
// In Windows, filename might have wildcards like "file.*" or "dir/*"
void Jidac::scandir(string filename) {

  // Don't scan diretories excluded by -not
  for (unsigned i=0; i<notfiles.size(); ++i)
    if (ispath(notfiles[i].c_str(), filename.c_str()))
      return;

#ifdef unix

  // Add regular files and directories
  while (filename.size()>1 && filename[filename.size()-1]=='/')
    filename=filename.substr(0, filename.size()-1);  // remove trailing /
  struct stat sb;
  if (!lstat(filename.c_str(), &sb)) {
    if (S_ISREG(sb.st_mode))
      addfile(filename, decimal_time(sb.st_mtime), sb.st_size,
              'u'+(sb.st_mode<<8));

    // Traverse directory
    if (S_ISDIR(sb.st_mode)) {
      addfile(filename=="/" ? "/" : filename+"/", decimal_time(sb.st_mtime),
              0, 'u'+(int64_t(sb.st_mode)<<8));
      DIR* dirp=opendir(filename.c_str());
      if (dirp) {
        for (dirent* dp=readdir(dirp); dp; dp=readdir(dirp)) {
          if (strcmp(".", dp->d_name) && strcmp("..", dp->d_name)) {
            string s=filename;
            if (s!="/") s+="/";
            s+=dp->d_name;
            scandir(s);
          }
        }
        closedir(dirp);
      }
      else
        perror(filename.c_str());
    }
  }
  else
    perror(filename.c_str());

#else  // Windows: expand wildcards in filename

  // Expand wildcards
  WIN32_FIND_DATA ffd;
  string t=filename;
  if (t.size()>0 && t[t.size()-1]=='/') t+="*";
  HANDLE h=FindFirstFile(utow(t.c_str()).c_str(), &ffd);
  if (h==INVALID_HANDLE_VALUE
      && GetLastError()!=ERROR_FILE_NOT_FOUND
      && GetLastError()!=ERROR_PATH_NOT_FOUND)
    printerr(t.c_str());
  while (h!=INVALID_HANDLE_VALUE) {

    // For each file, get name, date, size, attributes
    SYSTEMTIME st;
    int64_t edate=0;
    if (FileTimeToSystemTime(&ffd.ftLastWriteTime, &st))
      edate=st.wYear*10000000000LL+st.wMonth*100000000LL+st.wDay*1000000
            +st.wHour*10000+st.wMinute*100+st.wSecond;
    const int64_t esize=ffd.nFileSizeLow+(int64_t(ffd.nFileSizeHigh)<<32);
    const int64_t eattr='w'+(int64_t(ffd.dwFileAttributes)<<8);

    // Ignore links, the names "." and ".." or any unselected file
    t=wtou(ffd.cFileName);
    if (ffd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT
        || t=="." || t=="..") edate=0;  // don't add
    string fn=path(filename)+t;

    // Save directory names with a trailing / and scan their contents
    // Otherwise, save plain files
    if (edate) {
      if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) fn+="/";
      addfile(fn, edate, esize, eattr);
      if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        fn+="*";
        scandir(fn);
      }

      // enumerate alternate streams (Win2003/Vista or later)
      else if (findFirstStreamW && findNextStreamW) {
        WIN32_FIND_STREAM_DATA fsd;
        HANDLE ah=findFirstStreamW(utow(fn.c_str()).c_str(),
            FindStreamInfoStandard, &fsd, 0);
        while (ah!=INVALID_HANDLE_VALUE && findNextStreamW(ah, &fsd))
          addfile(fn+wtou(fsd.cStreamName), edate,
              fsd.StreamSize.QuadPart, eattr);
        if (ah!=INVALID_HANDLE_VALUE) FindClose(ah);
      }
    }
    if (!FindNextFile(h, &ffd)) {
      if (GetLastError()!=ERROR_NO_MORE_FILES) printerr(fn.c_str());
      break;
    }
  }
  FindClose(h);
#endif
}

// Add external file and its date, size, and attributes to dt
void Jidac::addfile(string filename, int64_t edate,
                    int64_t esize, int64_t eattr) {
  if (!isselected(filename.c_str(), false)) return;
  DT& d=edt[filename];
  d.date=edate;
  d.size=esize;
  d.attr=noattributes?0:eattr;
  d.data=0;
}

//////////////////////////////// add //////////////////////////////////

// Append n bytes of x to sb in LSB order
inline void puti(libzpaq::StringBuffer& sb, uint64_t x, int n) {
  for (; n>0; --n) sb.put(x&255), x>>=8;
}

// Print percent done (td/ts) and estimated time remaining
void print_progress(int64_t ts, int64_t td, int sum) {
  if (td>ts) td=ts;
  if (td>=1000000) {
    double eta=0.001*(mtime()-global_start)*(ts-td)/(td+1.0);
    printf("%5.2f%% %d:%02d:%02d ", td*100.0/(ts+0.5),
       int(eta/3600), int(eta/60)%60, int(eta)%60);
    if (sum>0) printf("\r"), fflush(stdout);
  }
}

// A CompressJob is a queue of blocks to compress and write to the archive.
// Each block cycles through states EMPTY, FILLING, FULL, COMPRESSING,
// COMPRESSED, WRITING. The main thread waits for EMPTY buffers and
// fills them. A set of compressThreads waits for FULL threads and compresses
// them. A writeThread waits for COMPRESSED buffers at the front
// of the queue and writes and removes them.

// Buffer queue element
struct CJ {
  enum {EMPTY, FULL, COMPRESSING, COMPRESSED, WRITING} state;
  StringBuffer in;       // uncompressed input
  StringBuffer out;      // compressed output
  string filename;       // to write in filename field
  string comment;        // if "" use default
  string method;         // compression level or "" to mark end of data
  Semaphore full;        // 1 if in is FULL of data ready to compress
  Semaphore compressed;  // 1 if out contains COMPRESSED data
  CJ(): state(EMPTY) {}
};

// Instructions to a compression job
class CompressJob {
public:
  Mutex mutex;           // protects state changes
private:
  int job;               // number of jobs
  CJ* q;                 // buffer queue
  unsigned qsize;        // number of elements in q
  int front;             // next to remove from queue
  libzpaq::Writer* out;  // archive
  Semaphore empty;       // number of empty buffers ready to fill
  Semaphore compressors; // number of compressors available to run
public:
  friend ThreadReturn compressThread(void* arg);
  friend ThreadReturn writeThread(void* arg);
  CompressJob(int threads, int buffers, libzpaq::Writer* f):
      job(0), q(0), qsize(buffers), front(0), out(f) {
    q=new CJ[buffers];
    if (!q) throw std::bad_alloc();
    init_mutex(mutex);
    empty.init(buffers);
    compressors.init(threads);
    for (int i=0; i<buffers; ++i) {
      q[i].full.init(0);
      q[i].compressed.init(0);
    }
  }
  ~CompressJob() {
    for (int i=qsize-1; i>=0; --i) {
      q[i].compressed.destroy();
      q[i].full.destroy();
    }
    compressors.destroy();
    empty.destroy();
    destroy_mutex(mutex);
    delete[] q;
  }      
  void write(StringBuffer& s, const char* filename, string method,
             const char* comment=0);
  vector<int> csize;  // compressed block sizes
};

// Write s at the back of the queue. Signal end of input with method=""
void CompressJob::write(StringBuffer& s, const char* fn, string method,
                        const char* comment) {
  for (unsigned k=(method=="")?qsize:1; k>0; --k) {
    empty.wait();
    lock(mutex);
    unsigned i, j;
    for (i=0; i<qsize; ++i) {
      if (q[j=(i+front)%qsize].state==CJ::EMPTY) {
        q[j].filename=fn?fn:"";
        q[j].comment=comment?comment:"jDC\x01";
        q[j].method=method;
        q[j].in.resize(0);
        q[j].in.swap(s);
        q[j].state=CJ::FULL;
        q[j].full.signal();
        break;
      }
    }
    release(mutex);
    assert(i<qsize);  // queue should not be full
  }
}

// Compress data in the background, one per buffer
ThreadReturn compressThread(void* arg) {
  CompressJob& job=*(CompressJob*)arg;
  int jobNumber=0;
  try {

    // Get job number = assigned position in queue
    lock(job.mutex);
    jobNumber=job.job++;
    assert(jobNumber>=0 && jobNumber<int(job.qsize));
    CJ& cj=job.q[jobNumber];
    release(job.mutex);

    // Work until done
    while (true) {
      cj.full.wait();
      lock(job.mutex);

      // Check for end of input
      if (cj.method=="") {
        cj.compressed.signal();
        release(job.mutex);
        return 0;
      }

      // Compress
      assert(cj.state==CJ::FULL);
      cj.state=CJ::COMPRESSING;
      release(job.mutex);
      job.compressors.wait();
      libzpaq::compressBlock(&cj.in, &cj.out, cj.method.c_str(),
          cj.filename.c_str(), cj.comment=="" ? 0 : cj.comment.c_str());
      cj.in.resize(0);
      lock(job.mutex);
      cj.state=CJ::COMPRESSED;
      cj.compressed.signal();
      job.compressors.signal();
      release(job.mutex);
    }
  }
  catch (std::exception& e) {
    lock(job.mutex);
    fflush(stdout);
    fprintf(stderr, "job %d: %s\n", jobNumber+1, e.what());
    release(job.mutex);
    exit(1);
  }
  return 0;
}

// Write compressed data to the archive in the background
ThreadReturn writeThread(void* arg) {
  CompressJob& job=*(CompressJob*)arg;
  try {

    // work until done
    while (true) {

      // wait for something to write
      CJ& cj=job.q[job.front];  // no other threads move front
      cj.compressed.wait();

      // Quit if end of input
      lock(job.mutex);
      if (cj.method=="") {
        release(job.mutex);
        return 0;
      }

      // Write to archive
      assert(cj.state==CJ::COMPRESSED);
      cj.state=CJ::WRITING;
      job.csize.push_back(cj.out.size());
      if (job.out && cj.out.size()>0) {
        release(job.mutex);
        assert(cj.out.c_str());
        const char* p=cj.out.c_str();
        int64_t n=cj.out.size();
        const int64_t N=1<<30;
        while (n>N) {
          job.out->write(p, N);
          p+=N;
          n-=N;
        }
        job.out->write(p, n);
        lock(job.mutex);
      }
      cj.out.resize(0);
      cj.state=CJ::EMPTY;
      job.front=(job.front+1)%job.qsize;
      job.empty.signal();
      release(job.mutex);
    }
  }
  catch (std::exception& e) {
    fflush(stdout);
    fprintf(stderr, "zpaq exiting from writeThread: %s\n", e.what());
    exit(1);
  }
  return 0;
}

// Write a ZPAQ compressed JIDAC block header. Output size should not
// depend on input data.
void writeJidacHeader(libzpaq::Writer *out, int64_t date,
                      int64_t cdata, unsigned htsize) {
  if (!out) return;
  assert(date>=19000000000000LL && date<30000000000000LL);
  StringBuffer is;
  puti(is, cdata, 8);
  libzpaq::compressBlock(&is, out, "0",
      ("jDC"+itos(date, 14)+"c"+itos(htsize, 10)).c_str(), "jDC\x01");
}

// Maps sha1 -> fragment ID in ht with known size
class HTIndex {
  vector<HT>& htr;  // reference to ht
  libzpaq::Array<unsigned> t;  // sha1 prefix -> index into ht
  unsigned htsize;  // number of IDs in t

  // Compuate a hash index for sha1[20]
  unsigned hash(const char* sha1) {
    return (*(const unsigned*)sha1)&(t.size()-1);
  }

public:
  // r = ht, sz = estimated number of fragments needed
  HTIndex(vector<HT>& r, size_t sz): htr(r), t(0), htsize(1) {
    int b;
    for (b=1; sz*3>>b; ++b);
    t.resize(1, b-1);
    update();
  }

  // Find sha1 in ht. Return its index or 0 if not found.
  unsigned find(const char* sha1) {
    unsigned h=hash(sha1);
    for (unsigned i=0; i<t.size(); ++i) {
      if (t[h^i]==0) return 0;
      if (memcmp(sha1, htr[t[h^i]].sha1, 20)==0) return t[h^i];
    }
    return 0;
  }

  // Update index of ht. Do not index if fragment size is unknown.
  void update() {
    char zero[20]={0};
    while (htsize<htr.size()) {
      if (htsize>=t.size()/4*3) {
        t.resize(t.size(), 1);
        htsize=1;
      }
      if (htr[htsize].usize>=0 && memcmp(htr[htsize].sha1, zero, 20)!=0) {
        unsigned h=hash((const char*)htr[htsize].sha1);
        for (unsigned i=0; i<t.size(); ++i) {
          if (t[h^i]==0) {
            t[h^i]=htsize;
            break;
          }
        }
      }
      ++htsize;
    }
  }    
};

// Sort by sortkey, then by full path
bool compareFilename(DTMap::iterator ap, DTMap::iterator bp) {
  if (ap->second.data!=bp->second.data)
    return ap->second.data<bp->second.data;
  return ap->first<bp->first;
}

// For writing to two archives at once
struct WriterPair: public libzpaq::Writer {
  OutputArchive *a, *b;
  void put(int c) {
    if (a) a->put(c);
    if (b) b->put(c);
  }
  void write(const char* buf, int n) {
    if (a) a->write(buf, n);
    if (b) b->write(buf, n);
  }
  WriterPair(): a(0), b(0) {}
};

// Add or delete files from archive. Return 1 if error else 0.
int Jidac::add() {

  // Read archive or index into ht, dt, ver.
  int errors=0;
  const bool archive_exists=exists(subpart(archive, 1).c_str());
  string arcname=archive;  // input archive name
  if (index) arcname=index;
  int64_t header_pos=0;
  if (exists(subpart(arcname, 1).c_str()))
    header_pos=read_archive(arcname.c_str(), &errors);

  // Set arcname, offset, header_pos, and salt to open out archive
  arcname=archive;  // output file name
  int64_t offset=0;  // total size of existing parts
  char salt[32]={0};  // encryption salt
  if (password) libzpaq::random(salt, 32);

  // Remote archive
  if (index) {
    if (dcsize>0) error("index is a regular archive");
    if (version!=DEFAULT_VERSION) error("cannot truncate with an index");
    offset=header_pos+dhsize;
    header_pos=32*(password && offset==0);
    arcname=subpart(archive, ver.size());
    if (exists(arcname.c_str())) {
      printUTF8(arcname.c_str(), stderr);
      fprintf(stderr, ": archive exists\n");
      error("archive exists");
    }
    if (password) {  // derive archive salt from index
      FP fp=fopen(index, RB);
      if (fp!=FPNULL) {
        if (fread(salt, 1, 32, fp)!=32) error("cannot read salt from index");
        salt[0]^='7'^'z';
        fclose(fp);
      }
    }
  }

  // Local single or multi-part archive
  else {
    int parts=0;  // number of existing parts in multipart
    string part0=subpart(archive, 0);
    if (part0!=archive) {  // multi-part?
      for (int i=1;; ++i) {
        string partname=subpart(archive, i);
        if (partname==part0) error("too many archive parts");
        FP fp=fopen(partname.c_str(), RB);
        if (fp==FPNULL) break;
        ++parts;
        fseeko(fp, 0, SEEK_END);
        offset+=ftello(fp);
        fclose(fp);
      }
      header_pos=32*(password && parts==0);
      arcname=subpart(archive, parts+1);
      if (exists(arcname.c_str())) error("part exists");
    }

    // Get salt from first part if it exists
    if (password) {
      FP fp=fopen(subpart(archive, 1).c_str(), RB);
      if (fp==FPNULL) {
        if (header_pos>32) error("archive first part not found");
        header_pos=32;
      }
      else {
        if (fread(salt, 1, 32, fp)!=32) error("cannot read salt");
        fclose(fp);
      }
    }
  }
  if (exists(arcname.c_str())) printf("Updating ");
  else printf("Creating ");
  printUTF8(arcname.c_str());
  printf(" at offset %1.0f + %1.0f\n", double(header_pos), double(offset));

  // Set method
  if (method=="") method="1";
  if (method.size()==1) {  // set default blocksize
    if (method[0]>='2' && method[0]<='9') method+="6";
    else method+="4";
  }
  if (strchr("0123456789xs", method[0])==0)
    error("-method must begin with 0..5, x, s");
  assert(method.size()>=2);
  if (method[0]=='s' && index) error("cannot index in streaming mode");

  // Set block and fragment sizes
  if (fragment<0) fragment=0;
  const int log_blocksize=20+atoi(method.c_str()+1);
  if (log_blocksize<20 || log_blocksize>31) error("blocksize must be 0..11");
  const unsigned blocksize=(1u<<log_blocksize)-4096;
  const unsigned MAX_FRAGMENT=fragment>19 || (8128u<<fragment)>blocksize-12
      ? blocksize-12 : 8128u<<fragment;
  const unsigned MIN_FRAGMENT=fragment>25 || (64u<<fragment)>MAX_FRAGMENT
      ? MAX_FRAGMENT : 64u<<fragment;

  // Don't mix streaming and journaling
  for (unsigned i=0; i<block.size(); ++i) {
    if (method[0]=='s') {
      if (block[i].usize>=0)
        error("cannot update journaling archive in streaming format");
    }
    else if (block[i].usize<0)
      error("cannot update streaming archive in journaling format");
  }

  // Make list of files to add or delete
  for (unsigned i=0; i<files.size(); ++i)
    scandir(files[i].c_str());

  // Sort the files to be added by filename extension and decreasing size
  vector<DTMap::iterator> vf;
  int64_t total_size=0;  // size of all input
  int64_t total_done=0;  // input deduped so far
  for (DTMap::iterator p=edt.begin(); p!=edt.end(); ++p) {
    DTMap::iterator a=dt.find(rename(p->first));
    if (a!=dt.end()) a->second.data=1;  // keep
    if (p->second.date && p->first!="" && p->first[p->first.size()-1]!='/'
        && (force || a==dt.end()
            || p->second.date!=a->second.date
            || p->second.size!=a->second.size)) {
      total_size+=p->second.size;

      // Key by first 5 bytes of filename extension, case insensitive
      int sp=0;  // sortkey byte position
      for (string::const_iterator q=p->first.begin(); q!=p->first.end(); ++q){
        uint64_t c=*q&255;
        if (c>='A' && c<='Z') c+='a'-'A';
        if (c=='/') sp=0, p->second.data=0;
        else if (c=='.') sp=8, p->second.data=0;
        else if (sp>3) p->second.data+=c<<(--sp*8);
      }

      // Key by descending size rounded to 16K
      int64_t s=p->second.size>>14;
      if (s>=(1<<24)) s=(1<<24)-1;
      p->second.data+=(1<<24)-s-1;
      vf.push_back(p);
    }
  }
  std::sort(vf.begin(), vf.end(), compareFilename);

  // Test for reliable access to archive
  if (archive_exists!=exists(subpart(archive, 1).c_str()))
    error("archive access is intermittent");

  // Open output
  OutputArchive out(arcname.c_str(), password, salt, offset);
  out.seek(header_pos, SEEK_SET);

  // Start compress and write jobs
  vector<ThreadID> tid(threads*2-1);
  ThreadID wid;
  CompressJob job(threads, tid.size(), &out);
  printf(
      "Adding %1.6f MB in %d files -method %s -threads %d at %s.\n",
      total_size/1000000.0, int(vf.size()), method.c_str(), threads,
      dateToString(date).c_str());
  for (unsigned i=0; i<tid.size(); ++i) run(tid[i], compressThread, &job);
  run(wid, writeThread, &job);

  // Append in streaming mode. Each file is a separate block. Large files
  // are split into blocks of size blocksize.
  int64_t dedupesize=0;  // input size after dedupe
  if (method[0]=='s') {
    StringBuffer sb(blocksize+4096-128);
    for (unsigned fi=0; fi<vf.size(); ++fi) {
      DTMap::iterator p=vf[fi];
      print_progress(total_size, total_done, summary);
      if (summary<=0) {
        printf("+ ");
        printUTF8(p->first.c_str());
        printf(" %1.0f\n", p->second.size+0.0);
      }
      FP in=fopen(p->first.c_str(), RB);
      if (in==FPNULL) {
        printerr(p->first.c_str());
        total_size-=p->second.size;
        ++errors;
        continue;
      }
      uint64_t i=0;
      const int BUFSIZE=4096;
      char buf[BUFSIZE];
      while (true) {
        int r=fread(buf, 1, BUFSIZE, in);
        sb.write(buf, r);
        i+=r;
        if (r==0 || sb.size()+BUFSIZE>blocksize) {
          string filename="";
          string comment="";
          if (i==sb.size()) {  // first block?
            filename=rename(p->first);
            comment=itos(p->second.date);
            if ((p->second.attr&255)>0) {
              comment+=" ";
              comment+=char(p->second.attr&255);
              comment+=itos(p->second.attr>>8);
            }
          }
          total_done+=sb.size();
          job.write(sb, filename.c_str(), method, comment.c_str());
          assert(sb.size()==0);
        }
        if (r==0) break;
      }
      fclose(in);
    }

    // Wait for jobs to finish
    job.write(sb, 0, "");  // signal end of input
    for (unsigned i=0; i<tid.size(); ++i) join(tid[i]);
    join(wid);

    // Done
    const int64_t outsize=out.tell();
    printf("%1.0f + (%1.0f -> %1.0f) = %1.0f\n",
        double(header_pos),
        double(total_size),
        double(outsize-header_pos),
        double(outsize));
    out.close();
    return errors>0;
  }  // end if streaming

  // Adjust date to maintain sequential order
  if (ver.size() && ver.back().lastdate>=date) {
    const int64_t newdate=decimal_time(unix_time(ver.back().lastdate)+1);
    fflush(stdout);
    fprintf(stderr, "Warning: adjusting date from %s to %s\n",
      dateToString(date).c_str(), dateToString(newdate).c_str());
    assert(newdate>date);
    date=newdate;
  }

  // Build htinv for fast lookups of sha1 in ht
  HTIndex htinv(ht, ht.size()+(total_size>>(10+fragment))+vf.size());
  const unsigned htsize=ht.size();  // fragments at start of update

  // reserve space for the header block
  writeJidacHeader(&out, date, -1, htsize);
  const int64_t header_end=out.tell();

  // Compress until end of last file
  assert(method!="");
  StringBuffer sb(blocksize+4096-128);  // block to compress
  unsigned frags=0;    // number of fragments in sb
  unsigned redundancy=0;  // estimated bytes that can be compressed out of sb
  unsigned text=0;     // number of fragents containing text
  unsigned exe=0;      // number of fragments containing x86 (exe, dll)
  const int ON=4;      // number of order-1 tables to save
  unsigned char o1prev[ON*256]={0};  // last ON order 1 predictions
  libzpaq::Array<char> fragbuf(MAX_FRAGMENT);
  vector<unsigned> blocklist;  // list of starting fragments

  // For each file to be added
  for (unsigned fi=0; fi<=vf.size(); ++fi) {
    FP in=FPNULL;
    const int BUFSIZE=4096;  // input buffer
    char buf[BUFSIZE];
    int bufptr=0, buflen=0;  // read pointer and limit
    if (fi<vf.size()) {
      assert(vf[fi]->second.ptr.size()==0);
      DTMap::iterator p=vf[fi];

      // Open input file
      bufptr=buflen=0;
      in=fopen(p->first.c_str(), RB);
      if (in==FPNULL) {  // skip if not found
        p->second.date=0;
        total_size-=p->second.size;
        printerr(p->first.c_str());
        ++errors;
        continue;
      }
      p->second.data=1;  // add
    }

    // Read fragments
    int64_t fsize=0;  // file size after dedupe
    for (unsigned fj=0; true; ++fj) {
      int64_t sz=0;  // fragment size;
      unsigned hits=0;  // correct prediction count
      int c=EOF;  // current byte
      unsigned htptr=0;  // fragment index
      char sha1result[20]={0};  // fragment hash
      unsigned char o1[256]={0};  // order 1 context -> predicted byte
      if (fi<vf.size()) {
        int c1=0;  // previous byte
        unsigned h=0;  // rolling hash for finding fragment boundaries
        libzpaq::SHA1 sha1;
        assert(in!=FPNULL);
        while (true) {
          if (bufptr>=buflen) bufptr=0, buflen=fread(buf, 1, BUFSIZE, in);
          if (bufptr>=buflen) c=EOF;
          else c=(unsigned char)buf[bufptr++];
          if (c!=EOF) {
            if (c==o1[c1]) h=(h+c+1)*314159265u, ++hits;
            else h=(h+c+1)*271828182u;
            o1[c1]=c;
            c1=c;
            sha1.put(c);
            fragbuf[sz++]=c;
          }
          if (c==EOF
              || sz>=MAX_FRAGMENT
              || (fragment<=22 && h<(1u<<(22-fragment)) && sz>=MIN_FRAGMENT))
            break;
        }
        assert(sz<=MAX_FRAGMENT);
        total_done+=sz;

        // Look for matching fragment
        assert(uint64_t(sz)==sha1.usize());
        memcpy(sha1result, sha1.result(), 20);
        htptr=htinv.find(sha1result);
      }  // end if fi<vf.size()

      if (htptr==0) {  // not matched or last block

        // Analyze fragment for redundancy, x86, text.
        // Test for text: letters, digits, '.' and ',' followed by spaces
        //   and no invalid UTF-8.
        // Test for exe: 139 (mov reg, r/m) in lots of contexts.
        // 4 tests for redundancy, measured as hits/sz. Take the highest of:
        //   1. Successful prediction count in o1.
        //   2. Non-uniform distribution in o1 (counted in o2).
        //   3. Fraction of zeros in o1 (bytes never seen).
        //   4. Fraction of matches between o1 and previous o1 (o1prev).
        int text1=0, exe1=0;
        int64_t h1=sz;
        unsigned char o1ct[256]={0};  // counts of bytes in o1
        static const unsigned char dt[256]={  // 32768/((i+1)*204)
          160,80,53,40,32,26,22,20,17,16,14,13,12,11,10,10,
            9, 8, 8, 8, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5,
            4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        for (int i=0; i<256; ++i) {
          if (o1ct[o1[i]]<255) h1-=(sz*dt[o1ct[o1[i]]++])>>15;
          if (o1[i]==' ' && (isalnum(i) || i=='.' || i==',')) ++text1;
          if (o1[i] && (i<9 || i==11 || i==12 || (i>=14 && i<=31) || i>=240))
            --text1;
          if (i>=192 && i<240 && o1[i] && (o1[i]<128 || o1[i]>=192))
            --text1;
          if (o1[i]==139) ++exe1;
        }
        text1=(text1>=3);
        exe1=(exe1>=5);
        if (sz>0) h1=h1*h1/sz; // Test 2: near 0 if random.
        unsigned h2=h1;
        if (h2>hits) hits=h2;
        h2=o1ct[0]*sz/256;  // Test 3: bytes never seen or that predict 0.
        if (h2>hits) hits=h2;
        h2=0;
        for (int i=0; i<256*ON; ++i)  // Test 4: compare to previous o1.
          h2+=o1prev[i]==o1[i&255];
        h2=h2*sz/(256*ON);
        if (h2>hits) hits=h2;
        if (hits>sz) hits=sz;

        // Start a new block if the current block is almost full, or at
        // the start of a file that won't fit or doesn't share mutual
        // information with the current block, or last file.
        bool newblock=false;
        if (frags>0 && fj==0 && fi<vf.size()) {
          const int64_t esize=vf[fi]->second.size;
          const int64_t newsize=sb.size()+esize+(esize>>14)+4096+frags*4;
          if (newsize>blocksize/4 && redundancy<sb.size()/128) newblock=true;
          if (newblock) {  // test for mutual information
            unsigned ct=0;
            for (unsigned i=0; i<256*ON; ++i)
              if (o1prev[i] && o1prev[i]==o1[i&255]) ++ct;
            if (ct>ON*2) newblock=false;
          }
          if (newsize>=blocksize) newblock=true;  // won't fit?
        }
        if (sb.size()+sz+80+frags*4>=blocksize) newblock=true; // full?
        if (fi==vf.size()) newblock=true;  // last file?
        if (frags<1) newblock=false;  // block is empty?

        // Pad sb with fragment size list, then compress
        if (newblock) {
          assert(frags>0);
          assert(frags<ht.size());
          for (unsigned i=ht.size()-frags; i<ht.size(); ++i)
            puti(sb, ht[i].usize, 4);  // list of frag sizes
          puti(sb, 0, 4); // omit first frag ID to make block movable
          puti(sb, frags, 4);  // number of frags
          string m=method;
          if (isdigit(method[0]))
            m+=","+itos(redundancy/(sb.size()/256+1))
                 +","+itos((exe>frags)*2+(text>frags));
          string fn="jDC"+itos(date, 14)+"d"+itos(ht.size()-frags, 10);
          print_progress(total_size, total_done, summary);
          if (summary<=0)
            printf("[%u..%u] %u -method %s\n",
                unsigned(ht.size())-frags, unsigned(ht.size())-1,
                unsigned(sb.size()), m.c_str());
          if (method[0]!='i')
            job.write(sb, fn.c_str(), m.c_str());
          else {  // index: don't compress data
            job.csize.push_back(sb.size());
            sb.resize(0);
          }
          assert(sb.size()==0);
          blocklist.push_back(ht.size()-frags);  // mark block start
          frags=redundancy=text=exe=0;
          memset(o1prev, 0, sizeof(o1prev));
        }

        // Append fragbuf to sb and update block statistics
        assert(sz==0 || fi<vf.size());
        sb.write(&fragbuf[0], sz);
        ++frags;
        redundancy+=hits;
        exe+=exe1*4;
        text+=text1*2;
        if (sz>=MIN_FRAGMENT) {
          memmove(o1prev, o1prev+256, 256*(ON-1));
          memcpy(o1prev+256*(ON-1), o1, 256);
        }
      }  // end if frag not matched or last block

      // Update HT and ptr list
      if (fi<vf.size()) {
        if (htptr==0) {
          htptr=ht.size();
          ht.push_back(HT(sha1result, sz));
          htinv.update();
          fsize+=sz;
        }
        vf[fi]->second.ptr.push_back(htptr);
      }
      if (c==EOF) break;
    }  // end for each fragment fj
    if (fi<vf.size()) {
      dedupesize+=fsize;
      DTMap::iterator p=vf[fi];
      print_progress(total_size, total_done, summary);
      if (summary<=0) {
        string newname=rename(p->first.c_str());
        DTMap::iterator a=dt.find(newname);
        if (a==dt.end() || a->second.date==0) printf("+ ");
        else printf("# ");
        printUTF8(p->first.c_str());
        if (newname!=p->first) {
          printf(" -> ");
          printUTF8(newname.c_str());
        }
        printf(" %1.0f", p->second.size+0.0);
        if (fsize!=p->second.size) printf(" -> %1.0f", fsize+0.0);
        printf("\n");
      }
      assert(in!=FPNULL);
      fclose(in);
      in=FPNULL;
    }
  }  // end for each file fi
  assert(sb.size()==0);

  // Wait for jobs to finish
  job.write(sb, 0, "");  // signal end of input
  for (unsigned i=0; i<tid.size(); ++i) join(tid[i]);
  join(wid);

  // Open index
  salt[0]^='7'^'z';
  OutputArchive outi(index ? index : "", password, salt, 0);
  WriterPair wp;
  wp.a=&out;
  if (index) wp.b=&outi;
  writeJidacHeader(&outi, date, 0, htsize);

  // Append compressed fragment tables to archive
  int64_t cdatasize=out.tell()-header_end;
  StringBuffer is;
  assert(blocklist.size()==job.csize.size());
  blocklist.push_back(ht.size());
  for (unsigned i=0; i<job.csize.size(); ++i) {
    if (blocklist[i]<blocklist[i+1]) {
      puti(is, job.csize[i], 4);  // compressed size of block
      for (unsigned j=blocklist[i]; j<blocklist[i+1]; ++j) {
        is.write((const char*)ht[j].sha1, 20);
        puti(is, ht[j].usize, 4);
      }
      libzpaq::compressBlock(&is, &wp, "0",
          ("jDC"+itos(date, 14)+"h"+itos(blocklist[i], 10)).c_str(),
          "jDC\x01");
      is.resize(0);
    }
  }

  // Delete from archive
  int dtcount=0;  // index block header name
  int removed=0;  // count
  for (DTMap::iterator p=dt.begin(); p!=dt.end(); ++p) {
    if (p->second.date && !p->second.data) {
      puti(is, 0, 8);
      is.write(p->first.c_str(), strlen(p->first.c_str()));
      is.put(0);
      if (summary<=0) {
        printf("- ");
        printUTF8(p->first.c_str());
        printf("\n");
      }
      ++removed;
      if (is.size()>16000) {
        libzpaq::compressBlock(&is, &wp, "1",
            ("jDC"+itos(date)+"i"+itos(++dtcount, 10)).c_str(), "jDC\x01");
        is.resize(0);
      }
    }
  }

  // Append compressed index to archive
  int added=0;  // count
  for (DTMap::iterator p=edt.begin();; ++p) {
    if (p!=edt.end()) {
      string filename=rename(p->first);
      DTMap::iterator a=dt.find(filename);
      if (p->second.date && (a==dt.end() // new file
         || a->second.date!=p->second.date  // date change
         || (a->second.attr && a->second.attr!=p->second.attr)  // attr ch.
         || a->second.size!=p->second.size  // size change
         || (p->second.data && a->second.ptr!=p->second.ptr))) { // content
        if (summary<=0 && p->second.data==0) {  // not compressed?
          if (a==dt.end() || a->second.date==0) printf("+ ");
          else printf("# ");
          printUTF8(p->first.c_str());
          if (filename!=p->first) {
            printf(" -> ");
            printUTF8(filename.c_str());
          }
          printf("\n");
        }
        ++added;
        puti(is, p->second.date, 8);
        is.write(filename.c_str(), strlen(filename.c_str()));
        is.put(0);
        if ((p->second.attr&255)=='u') {  // unix attributes
          puti(is, 3, 4);
          puti(is, p->second.attr, 3);
        }
        else if ((p->second.attr&255)=='w') {  // windows attributes
          puti(is, 5, 4);
          puti(is, p->second.attr, 5);
        }
        else puti(is, 0, 4);  // no attributes
        if (a==dt.end() || p->second.data) a=p;  // use new frag pointers
        puti(is, a->second.ptr.size(), 4);  // list of frag pointers
        for (unsigned i=0; i<a->second.ptr.size(); ++i)
          puti(is, a->second.ptr[i], 4);
      }
    }
    if (is.size()>16000 || (is.size()>0 && p==edt.end())) {
      libzpaq::compressBlock(&is, &wp, "1",
          ("jDC"+itos(date)+"i"+itos(++dtcount, 10)).c_str(), "jDC\x01");
      is.resize(0);
    }
    if (p==edt.end()) break;
  }
  printf("%d +added, %d -removed.\n", added, removed);
  assert(is.size()==0);

  // Back up and write the header
  outi.close();
  int64_t archive_end=out.tell();
  out.seek(header_pos, SEEK_SET);
  writeJidacHeader(&out, date, cdatasize, htsize);
  out.seek(0, SEEK_END);
  int64_t archive_size=out.tell();
  out.close();

  // Truncate empty update from archive (if not indexed)
  if (!index) {
    if (added+removed==0 && archive_end-header_pos==104) // no update
      archive_end=header_pos;
    if (archive_end<archive_size) {
      if (archive_end>0) {
        printf("truncating archive from %1.0f to %1.0f\n",
            double(archive_size), double(archive_end));
        if (truncate(arcname.c_str(), archive_end)) printerr(archive.c_str());
      }
      else if (archive_end==0) {
        if (delete_file(arcname.c_str())) {
          printf("deleted ");
          printUTF8(arcname.c_str());
          printf("\n");
        }
      }
    }
  }
  fflush(stdout);
  fprintf(stderr, "\n%1.6f + (%1.6f -> %1.6f -> %1.6f) = %1.6f MB\n",
      header_pos/1000000.0, total_size/1000000.0, dedupesize/1000000.0,
      (archive_end-header_pos)/1000000.0, archive_end/1000000.0);
  return errors>0;
}

/////////////////////////////// extract ///////////////////////////////

// Return true if the internal file p
// and external file contents are equal or neither exists.
// If filename is 0 then return true if it is possible to compare.
bool Jidac::equal(DTMap::const_iterator p, const char* filename) {

  // test if all fragment sizes and hashes exist
  if (filename==0) {
    static const char zero[20]={0};
    for (unsigned i=0; i<p->second.ptr.size(); ++i) {
      unsigned j=p->second.ptr[i];
      if (j<1 || j>=ht.size()
          || ht[j].usize<0 || !memcmp(ht[j].sha1, zero, 20))
        return false;
    }
    return true;
  }

  // internal or neither file exists
  if (p->second.date==0) return !exists(filename);

  // directories always match
  if (p->first!="" && p->first[p->first.size()-1]=='/')
    return exists(filename);

  // compare sizes
  FP in=fopen(filename, RB);
  if (in==FPNULL) return false;
  fseeko(in, 0, SEEK_END);
  if (ftello(in)!=p->second.size) return fclose(in), false;

  // compare hashes
  fseeko(in, 0, SEEK_SET);
  libzpaq::SHA1 sha1;
  const int BUFSIZE=4096;
  char buf[BUFSIZE];
  for (unsigned i=0; i<p->second.ptr.size(); ++i) {
    unsigned f=p->second.ptr[i];
    if (f<1 || f>=ht.size() || ht[f].usize<0) return fclose(in), false;
    for (int j=0; j<ht[f].usize;) {
      int n=ht[f].usize-j;
      if (n>BUFSIZE) n=BUFSIZE;
      int r=fread(buf, 1, n, in);
      if (r!=n) return fclose(in), false;
      sha1.write(buf, n);
      j+=n;
    }
    if (memcmp(sha1.result(), ht[f].sha1, 20)!=0) return fclose(in), false;
  }
  if (fread(buf, 1, BUFSIZE, in)!=0) return fclose(in), false;
  fclose(in);
  return true;
}

// An extract job is a set of blocks with at least one file pointing to them.
// Blocks are extracted in separate threads, set READY -> WORKING.
// A block is extracted to memory up to the last fragment that has a file
// pointing to it. Then the checksums are verified. Then for each file
// pointing to the block, each of the fragments that it points to within
// the block are written in order.

struct ExtractJob {         // list of jobs
  Mutex mutex;              // protects state
  Mutex write_mutex;        // protects writing to disk
  int job;                  // number of jobs started
  Jidac& jd;                // what to extract
  FP outf;                  // currently open output file
  DTMap::iterator lastdt;   // currently open output file name
  double maxMemory;         // largest memory used by any block (test mode)
  int64_t total_size;       // bytes to extract
  int64_t total_done;       // bytes extracted so far
  ExtractJob(Jidac& j): job(0), jd(j), outf(FPNULL), lastdt(j.dt.end()),
      maxMemory(0), total_size(0), total_done(0) {
    init_mutex(mutex);
    init_mutex(write_mutex);
  }
  ~ExtractJob() {
    destroy_mutex(mutex);
    destroy_mutex(write_mutex);
  }
};

// Decompress blocks in a job until none are READY
ThreadReturn decompressThread(void* arg) {
  ExtractJob& job=*(ExtractJob*)arg;
  int jobNumber=0;

  // Get job number
  lock(job.mutex);
  jobNumber=++job.job;
  release(job.mutex);

  // Open archive for reading
  InputArchive in(job.jd.archive.c_str(), job.jd.password);
  if (!in.isopen()) return 0;
  StringBuffer out;

  // Look for next READY job.
  int next=0;  // current job
  while (true) {
    lock(job.mutex);
    for (unsigned i=0; i<=job.jd.block.size(); ++i) {
      unsigned k=i+next;
      if (k>=job.jd.block.size()) k-=job.jd.block.size();
      if (i==job.jd.block.size()) {  // no more jobs?
        release(job.mutex);
        return 0;
      }
      Block& b=job.jd.block[k];
      if (b.state==Block::READY && b.size>0 && b.usize>=0) {
        b.state=Block::WORKING;
        release(job.mutex);
        next=k;
        break;
      }
    }
    Block& b=job.jd.block[next];

    // Get uncompressed size of block
    unsigned output_size=0;  // minimum size to decompress
    assert(b.start>0);
    for (unsigned j=0; j<b.size; ++j) {
      assert(b.start+j<job.jd.ht.size());
      assert(job.jd.ht[b.start+j].usize>=0);
      output_size+=job.jd.ht[b.start+j].usize;
    }

    // Decompress
    double mem=0;  // how much memory used to decompress
    try {
      assert(b.start>0);
      assert(b.start<job.jd.ht.size());
      assert(b.size>0);
      assert(b.start+b.size<=job.jd.ht.size());
      in.seek(b.offset, SEEK_SET);
      libzpaq::Decompresser d;
      d.setInput(&in);
      out.resize(0);
      assert(b.usize>=0);
      assert(b.usize<=0xffffffffu);
      out.setLimit(b.usize);
      d.setOutput(&out);
      if (!d.findBlock(&mem)) error("archive block not found");
      if (mem>job.maxMemory) job.maxMemory=mem;
      while (d.findFilename()) {
        d.readComment();
        while (out.size()<output_size && d.decompress(1<<14));
        lock(job.mutex);
        print_progress(job.total_size, job.total_done, job.jd.summary);
        if (job.jd.summary<=0)
          printf("[%d..%d] -> %1.0f\n", b.start, b.start+b.size-1,
              out.size()+0.0);
        release(job.mutex);
        if (out.size()>=output_size) break;
        d.readSegmentEnd();
      }
      if (out.size()<output_size) {
        lock(job.mutex);
        fflush(stdout);
        fprintf(stderr, "output [%d..%d] %d of %d bytes\n",
             b.start, b.start+b.size-1, int(out.size()), output_size);
        release(job.mutex);
        error("unexpected end of compressed data");
      }

      // Verify fragment checksums if present
      uint64_t q=0;  // fragment start
      libzpaq::SHA1 sha1;
      assert(b.extracted==0);
      for (unsigned j=b.start; j<b.start+b.size; ++j) {
        assert(j>0 && j<job.jd.ht.size());
        assert(job.jd.ht[j].usize>=0);
        assert(job.jd.ht[j].usize<=0x7fffffff);
        if (q+job.jd.ht[j].usize>out.size())
          error("Incomplete decompression");
        char sha1result[20];
        sha1.write(out.c_str()+q, job.jd.ht[j].usize);
        memcpy(sha1result, sha1.result(), 20);
        q+=job.jd.ht[j].usize;
        if (memcmp(sha1result, job.jd.ht[j].sha1, 20)) {
          lock(job.mutex);
          fflush(stdout);
          fprintf(stderr, "Job %d: fragment %u size %d checksum failed\n",
                 jobNumber, j, job.jd.ht[j].usize);
          release(job.mutex);
          error("bad checksum");
        }
        ++b.extracted;
      }
    }

    // If out of memory, let another thread try
    catch (std::bad_alloc& e) {
      lock(job.mutex);
      fflush(stdout);
      fprintf(stderr, "Job %d killed: %s\n", jobNumber, e.what());
      b.state=Block::READY;
      b.extracted=0;
      out.resize(0);
      release(job.mutex);
      return 0;
    }

    // Other errors: assume bad input
    catch (std::exception& e) {
      lock(job.mutex);
      fflush(stdout);
      fprintf(stderr, "Job %d: skipping [%u..%u] at %1.0f: %s\n",
              jobNumber, b.start+b.extracted, b.start+b.size-1,
              b.offset+0.0, e.what());
      release(job.mutex);
      continue;
    }

    // Write the files in dt that point to this block
    lock(job.write_mutex);
    for (unsigned ip=0; ip<b.files.size(); ++ip) {
      DTMap::iterator p=b.files[ip];
      if (p->second.date==0 || p->second.data<0
          || p->second.data>=int64_t(p->second.ptr.size()))
        continue;  // don't write

      // Look for pointers to this block
      const vector<unsigned>& ptr=p->second.ptr;
      int64_t offset=0;  // write offset
      for (unsigned j=0; j<ptr.size(); ++j) {
        if (ptr[j]<b.start || ptr[j]>=b.start+b.extracted) {
          offset+=job.jd.ht[ptr[j]].usize;
          continue;
        }

        // Close last opened file if different
        if (p!=job.lastdt) {
          if (job.outf!=FPNULL) {
            assert(job.lastdt!=job.jd.dt.end());
            assert(job.lastdt->second.date);
            assert(job.lastdt->second.data
                   <int64_t(job.lastdt->second.ptr.size()));
            fclose(job.outf);
            job.outf=FPNULL;
          }
          job.lastdt=job.jd.dt.end();
        }

        // Open file for output
        if (job.lastdt==job.jd.dt.end()) {
          string filename=job.jd.rename(p->first);
          assert(job.outf==FPNULL);
          if (p->second.data==0) {
            if (!job.jd.dotest) makepath(filename);
            if (job.jd.summary<=0) {
              lock(job.mutex);
              print_progress(job.total_size, job.total_done, job.jd.summary);
              if (job.jd.summary<=0) {
                printf("> ");
                printUTF8(filename.c_str());
                printf("\n");
              }
              release(job.mutex);
            }
            if (!job.jd.dotest) {
              job.outf=fopen(filename.c_str(), WB);
              if (job.outf==FPNULL) {
                lock(job.mutex);
                printerr(filename.c_str());
                release(job.mutex);
              }
#ifndef unix
              else if ((p->second.attr&0x200ff)==0x20000+'w') {  // sparse?
                DWORD br=0;
                if (!DeviceIoControl(job.outf, FSCTL_SET_SPARSE,
                    NULL, 0, NULL, 0, &br, NULL))  // set sparse attribute
                  printerr(filename.c_str());
              }
#endif
            }
          }
          else if (!job.jd.dotest)
            job.outf=fopen(filename.c_str(), RBPLUS);  // update existing file
          if (!job.jd.dotest && job.outf==FPNULL) break;  // skip errors
          job.lastdt=p;
          assert(job.jd.dotest || job.outf!=FPNULL);
        }
        assert(job.lastdt==p);

        // Find block offset of fragment
        uint64_t q=0;  // fragment offset from start of block
        for (unsigned k=b.start; k<ptr[j]; ++k) {
          assert(k>0);
          assert(k<job.jd.ht.size());
          if (job.jd.ht[k].usize<0) error("streaming fragment in file");
          assert(job.jd.ht[k].usize>=0);
          q+=job.jd.ht[k].usize;
        }
        assert(q+job.jd.ht[ptr[j]].usize<=out.size());

        // Combine consecutive fragments into a single write
        assert(offset>=0);
        ++p->second.data;
        uint64_t usize=job.jd.ht[ptr[j]].usize;
        assert(usize<=0x7fffffff);
        assert(b.start+b.size<=job.jd.ht.size());
        while (j+1<ptr.size() && ptr[j+1]==ptr[j]+1
               && ptr[j+1]<b.start+b.size
               && job.jd.ht[ptr[j+1]].usize>=0
               && usize+job.jd.ht[ptr[j+1]].usize<=0x7fffffff) {
          ++p->second.data;
          assert(p->second.data<=int64_t(ptr.size()));
          assert(job.jd.ht[ptr[j+1]].usize>=0);
          usize+=job.jd.ht[ptr[++j]].usize;
        }
        assert(usize<=0x7fffffff);
        assert(q+usize<=out.size());

        // Write the merged fragment unless they are all zeros and it
        // does not include the last fragment.
        uint64_t nz=q;  // first nonzero byte in fragments to be written
        while (nz<q+usize && out.c_str()[nz]==0) ++nz;
        if (!job.jd.dotest && (nz<q+usize || j+1==ptr.size())) {
          fseeko(job.outf, offset, SEEK_SET);
          fwrite(out.c_str()+q, 1, usize, job.outf);
        }
        offset+=usize;
        lock(job.mutex);
        job.total_done+=usize;
        release(job.mutex);

        // Close file. If this is the last fragment then set date and attr.
        // Do not set read-only attribute in Windows yet.
        if (p->second.data==int64_t(ptr.size())) {
          assert(p->second.date);
          assert(job.lastdt!=job.jd.dt.end());
          assert(job.jd.dotest || job.outf!=FPNULL);
          if (!job.jd.dotest) {
            assert(job.outf!=FPNULL);
            string fn=job.jd.rename(p->first);
            int64_t attr=p->second.attr;
            int64_t date=p->second.date;
            if ((p->second.attr&0x1ff)=='w'+256) attr=0;  // read-only?
            if (p->second.data!=int64_t(p->second.ptr.size()))
              date=attr=0;  // not last frag
            close(fn.c_str(), date, attr, job.outf);
            job.outf=FPNULL;
          }
          job.lastdt=job.jd.dt.end();
        }
      } // end for j
    } // end for ip

    // Last file
    release(job.write_mutex);
  } // end while true

  // Last block
  return 0;
}

// Streaming output destination
struct OutputFile: public libzpaq::Writer {
  FP f;
  void put(int c) {
    char ch=c;
    if (f!=FPNULL) fwrite(&ch, 1, 1, f);
  }
  void write(const char* buf, int n) {if (f!=FPNULL) fwrite(buf, 1, n, f);}
  OutputFile(FP out=FPNULL): f(out) {}
};

// Copy at most n bytes from in to out (default all). Return how many copied.
int64_t copy(libzpaq::Reader& in, libzpaq::Writer& out, uint64_t n=~0ull) {
  const unsigned BUFSIZE=4096;
  int64_t result=0;
  char buf[BUFSIZE];
  while (n>0) {
    int nc=n>BUFSIZE ? BUFSIZE : n;
    int nr=in.read(buf, nc);
    if (nr<1) break;
    out.write(buf, nr);
    result+=nr;
    n-=nr;
  }
  return result;
}

// Extract files from archive. If force is true then overwrite
// existing files and set the dates and attributes of exising directories.
// Otherwise create only new files and directories. Return 1 if error else 0.
int Jidac::extract() {

  // Encrypt or decrypt whole archive
  if (repack && all) {
    if (files.size()>0 || tofiles.size()>0 || onlyfiles.size()>0
        || noattributes || version!=DEFAULT_VERSION || method!="")
      error("-repack -all does not allow partial copy");
    InputArchive in(archive.c_str(), password);
    if (force) delete_file(repack);
    if (exists(repack)) error("output file exists");

    // Get key and salt
    char salt[32]={0};
    if (new_password) libzpaq::random(salt, 32);

    // Copy
    OutputArchive out(repack, new_password, salt, 0);
    copy(in, out);
    printUTF8(archive.c_str());
    printf(" %1.0f ", in.tell()+.0);
    printUTF8(repack);
    printf(" -> %1.0f\n", out.tell()+.0);
    out.close();
    return 0;
  }

  // Read archive
  const int64_t sz=read_archive(archive.c_str());
  if (sz<1) error("archive not found");

  // test blocks
  for (unsigned i=0; i<block.size(); ++i) {
    if (block[i].bsize<0) error("negative block size");
    if (block[i].start<1) error("block starts at fragment 0");
    if (block[i].start>=ht.size()) error("block start too high");
    if (i>0 && block[i].start<block[i-1].start) error("unordered frags");
    if (i>0 && block[i].start==block[i-1].start) error("empty block");
    if (i>0 && block[i].offset<block[i-1].offset+block[i-1].bsize)
      error("unordered blocks");
    if (i>0 && block[i-1].offset+block[i-1].bsize>block[i].offset)
      error("overlapping blocks");
  }

  // Create index instead of extract files
  if (index) {
    if (ver.size()<2) error("no journaling data");
    if (force) delete_file(index);
    if (exists(index)) error("index file exists");

    // Get salt
    char salt[32];
    if (ver[1].offset==32) {  // encrypted?
      FP fp=fopen(subpart(archive, 1).c_str(), RB);
      if (fp==FPNULL) error("cannot read part 1");
      if (fread(salt, 1, 32, fp)!=32) error("cannot read salt");
      salt[0]^='7'^'z';  // for index
      fclose(fp);
    }
    InputArchive in(archive.c_str(), password);
    OutputArchive out(index, password, salt, 0);
    for (unsigned i=1; i<ver.size(); ++i) {
      if (in.tell()!=ver[i].offset) error("I'm lost");

      // Read C block. Assume uncompressed and hash is present
      static char hdr[256]={0};  // Read C block
      int hsize=ver[i].data_offset-ver[i].offset;
      if (hsize<70 || hsize>255) error("bad C block size");
      if (in.read(hdr, hsize)!=hsize) error("EOF in header");
      if (hdr[hsize-36]!=9  // size of uncompressed block low byte
          || (hdr[hsize-22]&255)!=253  // start of SHA1 marker
          || (hdr[hsize-1]&255)!=255) {  // end of block marker
        for (int j=0; j<hsize; ++j)
          printf("%d%c", hdr[j]&255, j%10==9 ? '\n' : ' ');
        printf("at %1.0f\n", ver[i].offset+.0);
        error("C block in weird format");
      }
      memcpy(hdr+hsize-34, 
          "\x00\x00\x00\x00\x00\x00\x00\x00"  // csize = 0
          "\x00\x00\x00\x00"  // compressed data terminator
          "\xfd"  // start of hash marker
          "\x05\xfe\x40\x57\x53\x16\x6f\x12\x55\x59\xe7\xc9\xac\x55\x86"
          "\x54\xf1\x07\xc7\xe9"  // SHA-1('0'*8)
          "\xff", 34);  // EOB
      out.write(hdr, hsize);
      in.seek(ver[i].csize, SEEK_CUR);  // skip D blocks
      int64_t end=sz;
      if (i+1<ver.size()) end=ver[i+1].offset;
      int64_t n=end-in.tell();
      if (copy(in, out, n)!=n) error("EOF");  // copy H and I blocks
    }
    printUTF8(index);
    printf(" -> %1.0f\n", out.tell()+.0);
    out.close();
    return 0;
  }

  // Label files to extract with data=0.
  // Skip existing output files. If force then skip only if equal
  // and set date and attributes.
  ExtractJob job(*this);
  int total_files=0, skipped=0;
  for (DTMap::iterator p=dt.begin(); p!=dt.end(); ++p) {
    p->second.data=-1;  // skip
    if (p->second.date && p->first!="") {
      const string fn=rename(p->first);
      const bool isdir=p->first[p->first.size()-1]=='/';
      if (!repack && !dotest && force && !isdir && equal(p, fn.c_str())) {
        if (summary<=0) {  // identical
          printf("= ");
          printUTF8(fn.c_str());
          printf("\n");
        }
        close(fn.c_str(), p->second.date, p->second.attr);
        ++skipped;
      }
      else if (!repack && !dotest && !force && exists(fn)) {  // exists, skip
        if (summary<=0) {
          printf("? ");
          printUTF8(fn.c_str());
          printf("\n");
        }
        ++skipped;
      }
      else if (isdir)  // update directories later
        p->second.data=0;
      else if (block.size()>0) {  // files to decompress
        p->second.data=0;
        unsigned lo=0, hi=block.size()-1;  // block indexes for binary search
        for (unsigned i=0; p->second.data>=0 && i<p->second.ptr.size(); ++i) {
          unsigned j=p->second.ptr[i];  // fragment index
          if (j==0 || j>=ht.size() || ht[j].usize<-1) {
            fflush(stdout);
            printUTF8(p->first.c_str(), stderr);
            fprintf(stderr, ": bad frag IDs, skipping...\n");
            p->second.data=-1;  // skip
            continue;
          }
          assert(j>0 && j<ht.size());
          if (lo!=hi || lo>=block.size() || j<block[lo].start
              || (lo+1<block.size() && j>=block[lo+1].start)) {
            lo=0;  // find block with fragment j by binary search
            hi=block.size()-1;
            while (lo<hi) {
              unsigned mid=(lo+hi+1)/2;
              assert(mid>lo);
              assert(mid<=hi);
              if (j<block[mid].start) hi=mid-1;
              else (lo=mid);
            }
          }
          assert(lo==hi);
          assert(lo>=0 && lo<block.size());
          assert(j>=block[lo].start);
          assert(lo+1==block.size() || j<block[lo+1].start);
          unsigned c=j-block[lo].start+1;
          if (block[lo].size<c) block[lo].size=c;
          if (block[lo].files.size()==0 || block[lo].files.back()!=p)
            block[lo].files.push_back(p);
        }
        ++total_files;
        job.total_size+=p->second.size;
      }
    }  // end if selected
  }  // end for
  if (!force && skipped>0)
    printf("%d ?existing files skipped (-force overwrites).\n", skipped);
  if (force && skipped>0)
    printf("%d =identical files skipped.\n", skipped);

  // Repack to new archive
  if (repack) {

    // Get total D block size
    if (ver.size()<2) error("cannot repack streaming archive");
    int64_t csize=0;  // total compressed size of D blocks
    for (unsigned i=0; i<block.size(); ++i) {
      if (block[i].bsize<1) error("empty block");
      if (block[i].size>0) csize+=block[i].bsize;
    }

    // Open input
    InputArchive in(archive.c_str(), password);

    // Open output
    if (!force && exists(repack)) error("repack output exists");
    delete_file(repack);
    char salt[32]={0};
    if (new_password) libzpaq::random(salt, 32);
    OutputArchive out(repack, new_password, salt, 0);
    int64_t cstart=out.tell();

    // Write C block using first version date
    writeJidacHeader(&out, ver[1].date, -1, 1);
    int64_t dstart=out.tell();

    // Copy only referenced D blocks. If method then recompress.
    for (unsigned i=0; i<block.size(); ++i) {
      if (block[i].size>0) {
        in.seek(block[i].offset, SEEK_SET);
        copy(in, out, block[i].bsize);
      }
    }
    printf("Data %1.0f -> ", csize+.0);
    csize=out.tell()-dstart;
    printf("%1.0f\n", csize+.0);

    // Re-create referenced H blocks using latest date
    for (unsigned i=0; i<block.size(); ++i) {
      if (block[i].size>0) {
        StringBuffer is;
        puti(is, block[i].bsize, 4);
        for (unsigned j=0; j<block[i].frags; ++j) {
          const unsigned k=block[i].start+j;
          if (k<1 || k>=ht.size()) error("frag out of range");
          is.write((const char*)ht[k].sha1, 20);
          puti(is, ht[k].usize, 4);
        }
        libzpaq::compressBlock(&is, &out, "0",
            ("jDC"+itos(ver.back().date, 14)+"h"
            +itos(block[i].start, 10)).c_str(),
            "jDC\x01");
      }
    }

    // Append I blocks of selected files
    unsigned dtcount=0;
    StringBuffer is;
    for (DTMap::iterator p=dt.begin();; ++p) {
      if (p!=dt.end() && p->second.date>0 && p->second.data>=0) {
        string filename=rename(p->first);
        puti(is, p->second.date, 8);
        is.write(filename.c_str(), strlen(filename.c_str()));
        is.put(0);
        if ((p->second.attr&255)=='u') {  // unix attributes
          puti(is, 3, 4);
          puti(is, p->second.attr, 3);
        }
        else if ((p->second.attr&255)=='w') {  // windows attributes
          puti(is, 5, 4);
          puti(is, p->second.attr, 5);
        }
        else puti(is, 0, 4);  // no attributes
        puti(is, p->second.ptr.size(), 4);  // list of frag pointers
        for (unsigned i=0; i<p->second.ptr.size(); ++i)
          puti(is, p->second.ptr[i], 4);
      }
      if (is.size()>16000 || (is.size()>0 && p==dt.end())) {
        libzpaq::compressBlock(&is, &out, "1",
            ("jDC"+itos(ver.back().date)+"i"+itos(++dtcount, 10)).c_str(),
            "jDC\x01");
        is.resize(0);
      }
      if (p==dt.end()) break;
    }

    // Summarize result
    printUTF8(archive.c_str());
    printf(" %1.0f -> ", sz+.0);
    printUTF8(repack);
    printf(" %1.0f\n", out.tell()+.0);

    // Rewrite C block
    out.seek(cstart, SEEK_SET);
    writeJidacHeader(&out, ver[1].date, csize, 1);
    out.close();
    return 0;
  }

  // Decompress archive in parallel
  printf("Extracting %1.6f MB in %d files -threads %d\n",
      job.total_size/1000000.0, total_files, threads);
  vector<ThreadID> tid(threads);
  for (unsigned i=0; i<tid.size(); ++i) run(tid[i], decompressThread, &job);

  // Extract streaming files
  unsigned segments=0;  // count
  InputArchive in(archive.c_str(), password);
  if (in.isopen()) {
    FP outf=FPNULL;
    DTMap::iterator dtptr=dt.end();
    for (unsigned i=0; i<block.size(); ++i) {
      if (block[i].usize<0 && block[i].size>0) {
        Block& b=block[i];
        try {
          in.seek(b.offset, SEEK_SET);
          libzpaq::Decompresser d;
          d.setInput(&in);
          if (!d.findBlock()) error("block not found");
          StringWriter filename;
          for (unsigned j=0; j<b.size; ++j) {
            if (!d.findFilename(&filename)) error("segment not found");
            d.readComment();

            // Start of new output file
            if (filename.s!="" || segments==0) {
              unsigned k;
              for (k=0; k<b.files.size(); ++k) {  // find in dt
                if (b.files[k]->second.ptr.size()>0
                    && b.files[k]->second.ptr[0]==b.start+j
                    && b.files[k]->second.date>0
                    && b.files[k]->second.data==0)
                  break;
              }
              if (k<b.files.size()) {  // found new file
                if (outf!=FPNULL) fclose(outf);
                outf=FPNULL;
                string outname=rename(b.files[k]->first);
                dtptr=b.files[k];
                lock(job.mutex);
                if (summary<=0) {
                  printf("> ");
                  printUTF8(outname.c_str());
                  printf("\n");
                }
                if (!dotest) {
                  makepath(outname);
                  outf=fopen(outname.c_str(), WB);
                  if (outf==FPNULL) printerr(outname.c_str());
                }
                release(job.mutex);
              }
              else {  // end of file
                if (outf!=FPNULL) fclose(outf);
                outf=FPNULL;
                dtptr=dt.end();
              }
            }

            // Decompress segment
            libzpaq::SHA1 sha1;
            d.setSHA1(&sha1);
            OutputFile o(outf);
            d.setOutput(&o);
            d.decompress();

            // Verify checksum
            char sha1result[21];
            d.readSegmentEnd(sha1result);
            if (sha1result[0]==1) {
              if (memcmp(sha1result+1, sha1.result(), 20)!=0)
                error("checksum failed");
            }
            else if (sha1result[0]!=0)
              error("unknown checksum type");
            ++b.extracted;
            if (dtptr!=dt.end()) ++dtptr->second.data;
            filename.s="";
            ++segments;
          }
        }
        catch(std::exception& e) {
          lock(job.mutex);
          printf("Skipping block: %s\n", e.what());
          release(job.mutex);
        }
      }
    }
    if (outf!=FPNULL) fclose(outf);
  }
  if (segments>0) printf("%u streaming segments extracted\n", segments);

  // Wait for threads to finish
  for (unsigned i=0; i<tid.size(); ++i) join(tid[i]);

  // Create empty directories and set file dates and attributes
  if (!dotest) {
    for (DTMap::reverse_iterator p=dt.rbegin(); p!=dt.rend(); ++p) {
      if (p->second.data>=0 && p->second.date && p->first!="") {
        string s=rename(p->first);
        if (p->first[p->first.size()-1]=='/')
          makepath(s, p->second.date, p->second.attr);
        else if ((p->second.attr&0x1ff)=='w'+256)  // read-only?
          close(s.c_str(), 0, p->second.attr);
      }
    }
  }

  // Report failed extractions
  unsigned extracted=0, errors=0;
  for (DTMap::iterator p=dt.begin(); p!=dt.end(); ++p) {
    string fn=rename(p->first);
    if (p->second.data>=0 && p->second.date
        && fn!="" && fn[fn.size()-1]!='/') {
      ++extracted;
      if (p->second.ptr.size()!=unsigned(p->second.data)) {
        fflush(stdout);
        if (++errors==1)
          fprintf(stderr,
          "\nFailed (extracted/total fragments, file):\n");
        fprintf(stderr, "%u/%u ",
                int(p->second.data), int(p->second.ptr.size()));
        printUTF8(fn.c_str(), stderr);
        fprintf(stderr, "\n");
      }
    }
  }
  if (errors>0) {
    fflush(stdout);
    fprintf(stderr,
        "\nExtracted %u of %u files OK (%u errors)"
        " using %1.3f MB x %d threads\n",
        extracted-errors, extracted, errors, job.maxMemory/1000000,
        int(tid.size()));
  }
  return errors>0;
}

/////////////////////////////// list //////////////////////////////////

// Return p<q for sorting files by decreasing size, then fragment ID list
bool compareFragmentList(DTMap::const_iterator p, DTMap::const_iterator q) {
  if (p->second.size!=q->second.size) return p->second.size>q->second.size;
  if (p->second.ptr<q->second.ptr) return true;
  if (q->second.ptr<p->second.ptr) return false;
  if (p->second.data!=q->second.data) return p->second.data<q->second.data;
  return p->first<q->first;
}

// Return p<q for sort by name and comparison result
bool compareName(DTMap::const_iterator p, DTMap::const_iterator q) {
  if (p->first!=q->first) return p->first<q->first;
  return p->second.data<q->second.data;
}

// List contents
int Jidac::list() {

  // Read archive into dt, which may be "" for empty.
  int64_t csize=0;
  if (archive!="") csize=read_archive(archive.c_str());

  // Read external files into edt
  for (unsigned i=0; i<files.size(); ++i)
    scandir(files[i].c_str());
  if (files.size()) printf("%d external files.\n", int(edt.size()));
  printf("\n");

  // Compute directory sizes as the sum of their contents
  DTMap* dp[2]={&dt, &edt};
  for (int i=0; i<2; ++i) {
    for (DTMap::iterator p=dp[i]->begin(); p!=dp[i]->end(); ++p) {
      int len=p->first.size();
      if (len>0 && p->first[len]!='/') {
        for (int j=0; j<len; ++j) {
          if (p->first[j]=='/') {
            DTMap::iterator q=dp[i]->find(p->first.substr(0, j+1));
            if (q!=dp[i]->end())
              q->second.size+=p->second.size;
          }
        }
      }
    }
  }

  // Make list of files to list. List each external file preceded
  // by the matching internal file, if any. Then list any unmatched
  // internal files at the end.
  vector<DTMap::iterator> filelist;
  for (DTMap::iterator p=edt.begin(); p!=edt.end(); ++p) {
    DTMap::iterator a=dt.find(rename(p->first));
    if (a!=dt.end() && (all || a->second.date)) {
      a->second.data='-';
      filelist.push_back(a);
    }
    p->second.data='+';
    filelist.push_back(p);
  }
  for (DTMap::iterator a=dt.begin(); a!=dt.end(); ++a) {
    if (a->second.data!='-' && (all || a->second.date)) {
      a->second.data='-';
      filelist.push_back(a);
    }
  }

  // Sort
  if (summary>0)
    sort(filelist.begin(), filelist.end(), compareFragmentList);

  // List
  int64_t usize=0;
  unsigned matches=0, mismatches=0, internal=0, external=0,
           duplicates=0;  // counts
  for (unsigned fi=0;
       fi<filelist.size() && (summary<=0 || int(fi)<summary); ++fi) {
    DTMap::iterator p=filelist[fi];

    // Compare external files
    if (summary<=0 && p->second.data=='-' && fi+1<filelist.size()
        && filelist[fi+1]->second.data=='+') {
      DTMap::const_iterator p1=filelist[fi+1];
      if ((force && equal(p, p1->first.c_str()))
          || (!force && p->second.date==p1->second.date
              && p->second.size==p1->second.size
              && (!p->second.attr || !p1->second.attr
                  || p->second.attr==p1->second.attr))) {
        p->second.data='=';
        ++fi;
      }
      else
        p->second.data='#';
    }

    // Compare with previous file in summary
    if (summary>0 && fi>0 && p->second.date && p->first!=""
        && p->first[p->first.size()-1]!='/'
        && p->second.ptr.size()
        && filelist[fi-1]->second.ptr==p->second.ptr)
      p->second.data='^';

    if (p->second.data=='=') ++matches;
    if (p->second.data=='#') ++mismatches;
    if (p->second.data=='-') ++internal;
    if (p->second.data=='+') ++external;
    if (p->second.data=='^') ++duplicates;

    // List selected comparison results
    if (!strchr(nottype.c_str(), p->second.data)) {
      if (p->first!="" && p->first[p->first.size()-1]!='/')
        usize+=p->second.size;
      printf("%c %s %12.0f ", char(p->second.data),
          dateToString(p->second.date).c_str(), p->second.size+0.0);
      if (!noattributes)
        printf("%s ", attrToString(p->second.attr).c_str());
      printUTF8(p->first.c_str());
      if (summary<0) {  // frag pointers
        const vector<unsigned>& ptr=p->second.ptr;
        bool hyphen=false;
        for (int j=0; j<int(ptr.size()); ++j) {
          if (j==0 || j==int(ptr.size())-1 || ptr[j]!=ptr[j-1]+1
              || ptr[j]!=ptr[j+1]-1) {
            if (!hyphen) printf(" ");
            hyphen=false;
            printf("%d", ptr[j]);
          }
          else {
            if (!hyphen) printf("-");
            hyphen=true;
          }
        }
      }
      unsigned v;  // list version updates, deletes, compressed size
      if (all>0 && p->first.size()==all+1u && (v=atoi(p->first.c_str()))>0
          && v<ver.size()) {  // version info
        printf(" +%d -%d -> %1.0f", ver[v].updates, ver[v].deletes,
            (v+1<ver.size() ? ver[v+1].offset : csize)-ver[v].offset+0.0);
        if (summary<0)  // print fragment range
          printf(" %u-%u", ver[v].firstFragment,
              v+1<ver.size()?ver[v+1].firstFragment-1:unsigned(ht.size())-1);
      }
      printf("\n");
    }
  }  // end for i = each file version

  // Compute dedupe size
  int64_t ddsize=0, allsize=0;
  unsigned nfiles=0, nfrags=0, unknown_frags=0, refs=0;
  vector<bool> ref(ht.size());
  for (DTMap::const_iterator p=dt.begin(); p!=dt.end(); ++p) {
    if (p->second.date) {
      ++nfiles;
      for (unsigned j=0; j<p->second.ptr.size(); ++j) {
        unsigned k=p->second.ptr[j];
        if (k>0 && k<ht.size()) {
          ++refs;
          if (ht[k].usize>=0) allsize+=ht[k].usize;
          if (!ref[k]) {
            ref[k]=true;
            ++nfrags;
            if (ht[k].usize>=0) ddsize+=ht[k].usize;
            else ++unknown_frags;
          }
        }
      }
    }
  }

  // Print archive statistics
  printf("\n"
      "%1.6f MB of %1.6f MB (%d files) shown\n"
      "  -> %1.6f MB (%u refs to %u of %u frags) after dedupe\n"
      "  -> %1.6f MB compressed.\n",
       usize/1000000.0, allsize/1000000.0, nfiles, 
       ddsize/1000000.0, refs, nfrags, unsigned(ht.size())-1,
       (csize+dhsize-dcsize)/1000000.0);
  if (unknown_frags)
    printf("%d fragments have unknown size\n", unknown_frags);
  if (files.size())
    printf(
       "%d =same, %d #different, %d +external, %d -internal\n",
        matches, mismatches, external, internal);
  if (summary>0)
    printf("%d of largest %d files are ^duplicates\n",
        duplicates, summary);
  if (dhsize!=dcsize)  // index?
    printf("Note: %1.0f of %1.0f compressed bytes are in archive\n",
        dcsize+0.0, dhsize+0.0);
  return 0;
}

/////////////////////////////// main //////////////////////////////////

// Convert argv to UTF-8 and replace \ with /
#ifdef unix
int main(int argc, const char** argv) {
#else
#ifdef _MSC_VER
int wmain(int argc, LPWSTR* argw) {
#else
int main() {
  int argc=0;
  LPWSTR* argw=CommandLineToArgvW(GetCommandLine(), &argc);
#endif
  vector<string> args(argc);
  libzpaq::Array<const char*> argp(argc);
  for (int i=0; i<argc; ++i) {
    args[i]=wtou(argw[i]);
    argp[i]=args[i].c_str();
  }
  const char** argv=&argp[0];
#endif

  global_start=mtime();  // get start time
  int errorcode=0;
  try {
    Jidac jidac;
    errorcode=jidac.doCommand(argc, argv);
  }
  catch (std::exception& e) {
    fflush(stdout);
    fprintf(stderr, "zpaq error: %s\n", e.what());
    errorcode=2;
  }
  fflush(stdout);
  fprintf(stderr, "%1.3f seconds %s\n", (mtime()-global_start)/1000.0,
      errorcode>1 ? "(with errors)" :
      errorcode>0 ? "(with warnings)" : "(all OK)");
  return errorcode;
}
