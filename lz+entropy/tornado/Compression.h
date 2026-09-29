#ifndef FREEARC_COMPRESSION_H
#define FREEARC_COMPRESSION_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#include "Common.h"


#ifdef __cplusplus
extern "C" {
#endif

// Signature of files made by my utilities
#define BULAT_ZIGANSHIN_SIGNATURE 0x26351817

// Константы для удобной записи объёмов памяти
#define b_ (1u)
#define kb (1024*b_)
#define mb (1024*kb)
#define gb (1024*mb)
#define terabyte (1024*uint64(gb))

// Количество байт, которые должны читаться/записываться за один раз во всех упаковщиках
#define BUFFER_SIZE (256*kb)

// Количество байт, которые должны читаться/записываться за один раз в быстрых методах и при распаковке асимметричных алгоритмов
#define LARGE_BUFFER_SIZE (256*kb)

// Количество байт, которые должны читаться/записываться за один раз в очень быстрых методах (storing, tornado и тому подобное)
// Этот объём минимизирует потери на disk seek operations - при условии, что одновременно не происходит в/в в другом потоке ;)
#define HUGE_BUFFER_SIZE (8*mb)

// C какой частотой надо сообщать о прогрессе в упаковке/распаковке
#define PROGRESS_CHUNK_SIZE (64*kb)

// Дополнительные определения для удобства создания парсеров строк методов сжатия
#define COMPRESSION_METHODS_DELIMITER            '+'   /* Разделитель методов сжатия в строковом описании компрессора */
#define COMPRESSION_METHOD_PARAMETERS_DELIMITER  ':'   /* Разделитель параметров в строковом описании метода сжатия */
#define MAX_COMPRESSION_METHODS    1000        /* Должно быть не меньше числа методов сжатия, регистрируемых с помощью AddCompressionMethod */
#define MAX_PARAMETERS             200         /* Должно быть не меньше максимального кол-ва параметров (разделённых двоеточиями), которое может иметь метод сжатия */
#define MAX_COMPRESSOR_STRLEN      2048        /* Максимальная длина строки, описывающей компрессор */
#define MAX_METHOD_STRLEN          512         /* Максимальная длина строки, описывающей метод сжатия */
#define MAX_METHODS_IN_COMPRESSOR  100         /* Максимальное число методов в одном компрессоре */
#define MAX_EXTERNAL_COMPRESSOR_SECTION_LENGTH 2048  /* Максимальная длина секции [External compressor] */


// ****************************************************************************************************************************
// ХЕЛПЕРЫ ЧТЕНИЯ/ЗАПИСИ ДАННЫХ В МЕТОДАХ СЖАТИЯ ******************************************************************************
// ****************************************************************************************************************************

// Тип функции для обратных вызовов
typedef int CALLBACK_FUNC (const char *what, void *data, int size, void *auxdata);

// Макросы для чтения/записи в(ы)ходных потоков с проверкой, что передано ровно столько данных, сколько было запрошено
#define checked_read(ptr,size)         {if ((x = callback("read" ,ptr,size,auxdata)) != size) {x>=0 && (x=FREEARC_ERRCODE_READ);  goto finished;}}
#define checked_write(ptr,size)        {if ((x = callback("write",ptr,size,auxdata)) != size) {x>=0 && (x=FREEARC_ERRCODE_WRITE); goto finished;}}
// Макрос для чтения входных потоков с проверкой на ошибки и конец входных данных
#define checked_eof_read(ptr,size)     {if ((x = callback("read", ptr,size,auxdata)) != size) {x>0  && (x=FREEARC_ERRCODE_READ);  goto finished;}}

// Auxiliary code to read/write data blocks and 4-byte headers
#define INIT() callback ("init", NULL, 0, auxdata)
#define DONE() callback ("done", NULL, 0, auxdata)

#define MALLOC(type, ptr, size)                                            \
{                                                                          \
    (ptr) = (type*) malloc ((size) * sizeof(type));                        \
    if ((ptr) == NULL) {                                                   \
        errcode = FREEARC_ERRCODE_NOT_ENOUGH_MEMORY;                       \
        goto finished;                                                     \
    }                                                                      \
}

#define BIGALLOC(type, ptr, size)                                          \
{                                                                          \
    (ptr) = (type*) BigAlloc ((size) * sizeof(type));                      \
    if ((ptr) == NULL) {                                                   \
        errcode = FREEARC_ERRCODE_NOT_ENOUGH_MEMORY;                       \
        goto finished;                                                     \
    }                                                                      \
}

#define READ(buf, size)                                                    \
{                                                                          \
    void *localBuf = (buf);                                                \
    int localSize  = (size);                                               \
    int localErrCode;                                                                                \
    if (localSize  &&  (localErrCode=callback("read",localBuf,localSize,auxdata)) != localSize) {    \
        errcode = localErrCode<0? localErrCode : FREEARC_ERRCODE_READ;                               \
        goto finished;                                                     \
    }                                                                      \
}

#define READ_LEN(len, buf, size)                                           \
{                                                                          \
    int localErrCode;                                                      \
    if ((localErrCode=(len)=callback("read",buf,size,auxdata)) < 0) {      \
        errcode = localErrCode;                                            \
        goto finished;                                                     \
    }                                                                      \
}

#define READ_LEN_OR_EOF(len, buf, size)                                    \
{                                                                          \
    int localErrCode;                                                      \
    if ((localErrCode=(len)=callback("read",buf,size,auxdata)) <= 0) {     \
        errcode = localErrCode;                                            \
        goto finished;                                                     \
    }                                                                      \
}

#define WRITE(buf, size)                                                   \
{                                                                          \
    void *localBuf = (buf);                                                \
    int localSize  = (size);                                               \
    int localErrCode;                                                                   \
    /* "write" callback on success guarantees to write all the data and may return 0 */ \
    if (localSize && (localErrCode=callback("write",localBuf,localSize,auxdata))<0) {   \
        errcode = localErrCode;                                                         \
        goto finished;                                                     \
    }                                                                      \
}

#define READ4(var)                                                         \
{                                                                          \
    unsigned char localHeader[4];                                          \
    READ (localHeader, 4);                                                 \
    (var) = value32 (localHeader);                                         \
}

#define READ4_OR_EOF(var)                                                  \
{                                                                          \
    int localHeaderSize;                                                   \
    unsigned char localHeader[4];                                          \
    READ_LEN_OR_EOF (localHeaderSize, localHeader, 4);                     \
    if (localHeaderSize!=4)  {errcode=FREEARC_ERRCODE_READ; goto finished;}\
    (var) = value32 (localHeader);                                         \
}

#define WRITE4(value)                                                      \
{                                                                          \
    unsigned char localHeader[4];                                          \
    setvalue32 (localHeader, value);                                       \
    WRITE (localHeader, 4);                                                \
}

#define QUASIWRITE(size)                                                   \
{                                                                          \
    int64 localSize = (size);                                              \
    callback ("quasiwrite", &localSize, (size), auxdata);                  \
}

#define PROGRESS(insize,outsize)                                           \
{                                                                          \
    int64 localSize[2] = {(int64)(insize),(int64)(outsize)};                             \
    callback ("progress", localSize, 0, auxdata);                          \
}

#define ReturnErrorCode(err)                                               \
{                                                                          \
    errcode = (err);                                                       \
    goto finished;                                                         \
}                                                                          \


// Buffered data output
#ifndef FREEARC_STANDALONE_TORNADO
#define FOPEN()   Buffer fbuffer(BUFFER_SIZE)
#define FWRITE(buf, size)                                                  \
{                                                                          \
    void *flocalBuf = (buf);                                               \
    int flocalSize = (size);                                               \
    int rem = fbuffer.remainingSpace();                                    \
    if (flocalSize>=4096) {                                                \
        FFLUSH();                                                          \
        WRITE(flocalBuf, flocalSize);                                      \
    } else if (flocalSize < rem) {                                         \
        fbuffer.put (flocalBuf, flocalSize);                               \
    } else {                                                               \
        fbuffer.put (flocalBuf, rem);                                      \
        FFLUSH();                                                          \
        fbuffer.put ((byte*)flocalBuf+rem, flocalSize-rem);                \
    }                                                                      \
}
#define FWRITESZ(value)                                                    \
{                                                                          \
    const char *flocalValue = (value);                                     \
    int flocalBytes = strlen(flocalValue) + 1;                             \
    FWRITE ((void*)flocalValue, flocalBytes);                              \
}
#define FWRITE4(value)                                                     \
{                                                                          \
    unsigned char flocalHeader[4];                                         \
    setvalue32 (flocalHeader, value);                                      \
    FWRITE (flocalHeader, 4);                                              \
}
#define FWRITE1(value)                                                     \
{                                                                          \
    unsigned char flocalHeader = (value);                                  \
    FWRITE (&flocalHeader, 1);                                             \
}
#define FFLUSH()  { WRITE (fbuffer.buf, fbuffer.len());  fbuffer.empty(); }
#define FCLOSE()  { FFLUSH(); }


// Буфер, используемый для организации нескольких независимых потоков записи
// в программе. Буфер умеет записывать в себя 8/16/32-разрядные числа и расширяться
// при необходимости. Позднее содержимое буфера сбрасывается в выходной поток.
// Дополнительно буфер поддерживает чтение ранее записанных в него данных.
// Конец записанной части буфера - это max(p,end), где p - текущий указатель,
// а end - максимальная позиция ранее записанных данных.
struct Buffer
{
    byte*  buf;              // адрес выделенного буфера
    byte*  p;                // текущий указатель чтения/записи внутри этого буфера
    byte*  end;              // адрес после конца прочитанных/записанных данных
    byte*  bufend;           // конец самого буфера

    Buffer (uint size=64*kb) { buf=p=end= (byte*) malloc(size);  bufend=buf+size; }
    ~Buffer()                { freebuf(); }
    void   freebuf()         { free(buf);  buf=p=end=NULL; }
    void   empty()           { p=end=buf; }
    int    len()             { return mymax(p,end)-buf; }

    void   put8 (uint x)     { reserve(sizeof(uint8 ));  *(uint8 *)p=x;    p+=sizeof(uint8 ); }
    void   put16(uint x)     { reserve(sizeof(uint16));  setvalue16(p,x);  p+=sizeof(uint16); }
    void   put32(uint x)     { reserve(sizeof(uint32));  setvalue32(p,x);  p+=sizeof(uint32); }

    void   put(void *b, int n)  { reserve(n);  memcpy(p,b,n);  p+=n; }
    void   puts (char *s)    { put (s, strlen(s)); }
    void   putsz(char *s)    { put (s, strlen(s)+1); }

    int    remainingSpace()  { return bufend-p; }
    void   reserve(uint n)   {
                               if (remainingSpace() < n)
                               {
                                 uint newsize = mymax(p+n-buf, (bufend-buf)*2);
                                 byte* newbuf = (byte*) realloc (buf, newsize);
                                 bufend = newbuf + newsize;
                                 p   += newbuf-buf;
                                 end += newbuf-buf;
                                 buf  = newbuf;
                               }
                             }

    void reverseBytes()      {
                               byte *lo = buf,  *hi = buf + len() - 1,  swap;
                               while (lo < hi)  { swap = *lo;  *lo++ = *hi;  *hi-- = swap; }
                             }
// Для чтения данных
    void   rewind()          { end=mymax(p,end);  p=buf; }
    uint   get8 ()           { uint x = *(uint8 *)p;  p+=sizeof(uint8 );  return x; }
    uint   get16()           { uint x = value16(p);   p+=sizeof(uint16);  return x; }
    uint   get32()           { uint x = value32(p);   p+=sizeof(uint32);  return x; }
    int    get(void *b, int n)  { n = mymin(remainingData(), n);  memcpy(b,p,n);  p+=n;  return n;}
    int    remainingData()   { return p<end? end-p : 0; }
    bool   eof()             { return remainingData()==0; }
};

#endif // !FREEARC_STANDALONE_TORNADO


// ****************************************************************************************************************************
// ВЫЧИСЛЕНИЕ CRC-32                                                                                                          *
// ****************************************************************************************************************************

#define INIT_CRC 0xffffffff

uint32 UpdateCRC (const void *Addr, size_t Size, uint32 StartCRC);     // Обновить CRC содержимым блока данных
uint32 CalcCRC   (const void *Addr, size_t Size);                      // Вычислить CRC блока данных


// ****************************************************************************************************************************
// УТИЛИТЫ ********************************************************************************************************************
// ****************************************************************************************************************************

// Параметр алгоритма сжатия/шифрования
typedef char *CPARAM;

// Алгоритм сжатия/шифрования, представленный в виде строки
typedef char *CMETHOD;

// Последовательность алгоритмов сжатия/шифрования, представленная в виде "exe+rep+lzma+aes"
typedef char *COMPRESSOR;

// Запросить сервис what метода сжатия method
int CompressionService (char *method, char *what, DEFAULT(int param,0), DEFAULT(void *data,NULL), DEFAULT(CALLBACK_FUNC *callback,NULL));

// Проверить, что данный компрессор включает алгоритм шифрования
int compressorIsEncrypted (COMPRESSOR c);
// Вычислить, сколько памяти нужно для распаковки данных, сжатых этим компрессором
MemSize compressorGetDecompressionMem (COMPRESSOR c);

// Get/set number of threads used for (de)compression
int  __cdecl GetCompressionThreads (void);
void __cdecl SetCompressionThreads (int threads);

// Used in 4x4 only: read entire input buffer before compression begins, allocate output buffer large enough to hold entire compressed output
extern int compress_all_at_once;
void __cdecl Set_compress_all_at_once (int n);
struct Set_compress_all_at_once_Until_end_of_block
{
  int save;
  Set_compress_all_at_once_Until_end_of_block (int n)  {save = compress_all_at_once;  Set_compress_all_at_once(n);}
  ~Set_compress_all_at_once_Until_end_of_block()       {Set_compress_all_at_once(save);}
};

// Enable debugging output
extern int debug_mode;
void __cdecl Set_debug_mode (int n);

// Load accelerated function either from facompress.dll or facompress_mt.dll
FARPROC LoadFromDLL (char *funcname, DEFAULT(int only_facompress_mt, FALSE));

// Other compression methods may chain-redefine this callback in order to perform their own cleanup procedures
extern void (*BeforeUnloadDLL)();

// This function unloads DLLs containing accelerated compression functions
void UnloadDLL (void);

#ifdef FREEARC_WIN
extern HINSTANCE hinstUnarcDll;   // unarc.dll instance
#endif

// This function should cleanup Compression Library
void compressionLib_cleanup (void);


// ****************************************************************************************************************************
// СЕРВИСЫ СЖАТИЯ И РАСПАКОВКИ ДАННЫХ *****************************************************************************************
// ****************************************************************************************************************************

enum COMPRESSION {COMPRESS, DECOMPRESS};  // Direction of operation

// Распаковать данные, упакованные заданным методом или цепочкой методов
int Decompress (char *method, CALLBACK_FUNC *callback, void *auxdata);
// Прочитать из входного потока обозначение метода сжатия и распаковать данные этим методом
int DecompressWithHeader (CALLBACK_FUNC *callback, void *auxdata);
// Распаковать данные в памяти, записав в выходной буфер не более outputSize байт.
// Возвращает код ошибки или количество байт, записанных в выходной буфер
int DecompressMem (char *method, void *input, int inputSize, void *output, int outputSize);
int DecompressMemWithHeader     (void *input, int inputSize, void *output, int outputSize);

#ifndef FREEARC_DECOMPRESS_ONLY
// Упаковать данные заданным методом или цепочкой методов
int Compress   (char *method, CALLBACK_FUNC *callback, void *auxdata);
// Записать в выходной поток обозначение метода сжатия и упаковать данные этим методом
int CompressWithHeader (char *method, CALLBACK_FUNC *callback, void *auxdata);
// Упаковать данные в памяти, записав в выходной буфер не более outputSize байт.
// Возвращает код ошибки или количество байт, записанных в выходной буфер
int CompressMem           (char *method, void *input, int inputSize, void *output, int outputSize);
int CompressMemWithHeader (char *method, void *input, int inputSize, void *output, int outputSize);
// Информация о памяти, необходимой для упаковки/распаковки, размере словаря и размере блока.
MemSize GetCompressionMem      (char *method);
MemSize GetMinCompressionMem   (char *method);
MemSize GetMinDecompressionMem (char *method);
// Возвратить в out_method новый метод сжатия, настроенный на использование
// соответствующего количества памяти/словаря/размера блока
int SetCompressionMem          (char *in_method, MemSize mem,  char *out_method);
int SetMinDecompressionMem     (char *in_method, MemSize mem,  char *out_method);
int SetDictionary              (char *in_method, MemSize dict, char *out_method);
int SetBlockSize               (char *in_method, MemSize bs,   char *out_method);
// Возвратить в out_method новый метод сжатия, уменьшив, если необходимо,
// используемую алгоритмом память / его словарь / размер блока
int LimitCompressionMem        (char *in_method, MemSize mem,  char *out_method);
int LimitMinDecompressionMem   (char *in_method, MemSize mem,  char *out_method);
int LimitDictionary            (char *in_method, MemSize dict, char *out_method);
int LimitBlockSize             (char *in_method, MemSize bs,   char *out_method);
#endif
MemSize GetDictionary          (char *method);
MemSize GetBlockSize           (char *method);
MemSize GetDecompressionMem    (char *method);
int     SetDecompressionMem    (char *in_method, MemSize mem,  char *out_method);
int     LimitDecompressionMem  (char *in_method, MemSize mem,  char *out_method);

// Вывести в out_method каноническое представление метода сжатия in_method (выполнить ParseCompressionMethod + ShowCompressionMethod)
//   purify!=0: подготовить представление method к записи в архив (например, убрать :t:i для 4x4)
int CanonizeCompressionMethod (char *in_method, char *out_method, int purify);

// Функция "(рас)паковки", копирующая данные один в один
int copy_data   (CALLBACK_FUNC *callback, void *auxdata);


// ****************************************************************************************************************************
// КЛАСС, РЕАЛИЗУЮЩИЙ ИНТЕРФЕЙС К МЕТОДУ СЖАТИЯ *******************************************************************************
// ****************************************************************************************************************************

#ifdef __cplusplus

// Абстрактный интерфейс к произвольному методу сжатия
class COMPRESSION_METHOD
{
public:
  // Функции распаковки и упаковки
  //   DeCompressMem can either compress or decompress, either from memory block `input` to `output` or calling `callback` for I/O.
  //   CodecState, unless NULL, points to the place for storing pointer to persistent codec state, such as allocated buffers
  //     and precomputed tables, that should be finally freed by the empty DeCompressMem() call.
  virtual int DeCompressMem (COMPRESSION direction, void *input, int inputSize, void *output, int *outputSize, CALLBACK_FUNC *callback=0, void *auxdata=0, void **CodecState=0);
  virtual int decompress (CALLBACK_FUNC *callback, void *auxdata) = 0;
#ifndef FREEARC_DECOMPRESS_ONLY
  virtual int compress   (CALLBACK_FUNC *callback, void *auxdata) = 0;

  // Информация о памяти, необходимой для упаковки/распаковки (Min - при :t1:i0, т.е. минимальном числе тредов/буферов - для ArcInfo и т.п.),
  // размере словаря (то есть насколько далеко заглядывает алгоритм в поиске похожих данных - для lz/bs схем),
  // и размере блока (то есть сколько максимум данных имеет смысл помещать в один солид-блок - для bs схем и lzp)
  virtual MemSize GetCompressionMem        (void)         = 0;
  virtual MemSize GetMinCompressionMem     (void)               {return GetCompressionMem();}
  virtual MemSize GetMinDecompressionMem   (void)               {return GetDecompressionMem();}
  // Настроить метод сжатия на использование заданного кол-ва памяти, словаря или размера блока
  virtual void    SetDictionary            (MemSize dict)       {}
  virtual void    SetBlockSize             (MemSize bs)         {}
  virtual void    SetCompressionMem        (MemSize mem)  = 0;
  virtual void    SetMinDecompressionMem   (MemSize mem)  = 0;  // для -ld при упаковке (т.е. при :t1:i0): настраиваем минимальный объём памяти, требуемый для распаковки
  // Ограничить используемую при упаковке/распаковке память, или словарь / размер блока
  virtual void    LimitDictionary          (MemSize dict)       {if (dict>0 && GetDictionary()          > dict)  SetDictionary(dict);}
  virtual void    LimitBlockSize           (MemSize bs)         {if (bs>0   && GetBlockSize()           > bs)    SetBlockSize(bs);}
  virtual void    LimitCompressionMem      (MemSize mem)        {if (mem>0  && GetCompressionMem()      > mem)   SetCompressionMem(mem);}
  virtual void    LimitMinDecompressionMem (MemSize mem)        {if (mem>0  && GetMinDecompressionMem() > mem)   SetMinDecompressionMem(mem);}
#endif
  virtual MemSize GetDictionary            (void)               {return 0;}
  virtual MemSize GetBlockSize             (void)               {return 0;}
  virtual MemSize GetAlgoMem               (void);                            // Объём памяти, характеризующий алгоритм
  virtual MemSize GetDecompressionMem      (void)         = 0;
  virtual void    SetDecompressionMem      (MemSize mem)        {}    // для -ld при распаковке (т.е. меняем только параметры типа :t:i, сохраняя совместимость с упакованными данными)
  virtual void    LimitDecompressionMem    (MemSize mem)        {if (mem>0  && GetDecompressionMem() > mem)   SetDecompressionMem(mem);}

  // Maximum possible inflation of incompressible input data
  virtual LongMemSize GetMaxCompressedSize (LongMemSize insize) {return insize + (insize/4) + 16*kb;}

  // Записать в buf[MAX_METHOD_STRLEN] строку, описывающую метод сжатия и его параметры (функция, обратная к ParseCompressionMethod)
  virtual void ShowCompressionMethod (char *buf, bool purify) = 0;

  // Универсальный метод. Параметры:
  //   what: "compress", "decompress", "setCompressionMem", "limitDictionary"...
  //   data: данные для операции в формате, зависящем от конкретной выполняемой операции
  //   param&result: простой числовой параметр, что достаточно для многих информационных операций
  // Неиспользуемые параметры устанавливаются в NULL/0. result<0 - код ошибки
  virtual int doit (char *what, int param, void *data, CALLBACK_FUNC *callback);

  // Check boolean method property
  bool is (char *request)   {return doit (request, 0, NULL, NULL) > 0;}

  double addtime;  // Дополнительное время, потраченное на сжатие (во внешних программах, дополнительных threads и т.д.)
  COMPRESSION_METHOD() {addtime=0;}
  virtual ~COMPRESSION_METHOD() {}
//  Debugging code:  char buf[100]; ShowCompressionMethod(buf,FALSE); printf("%s : %u => %u\n", buf, GetCompressionMem(), mem);
};


// ****************************************************************************************************************************
// ФАБРИКА COMPRESSION_METHOD *************************************************************************************************
// ****************************************************************************************************************************

// Сконструировать объект класса - наследника COMPRESSION_METHOD,
// реализующий метод сжатия, заданный в виде строки `method`
COMPRESSION_METHOD *ParseCompressionMethod (char* method);

typedef COMPRESSION_METHOD* (*CM_PARSER) (char** parameters);
typedef COMPRESSION_METHOD* (*CM_PARSER2) (char** parameters, void *data);
int AddCompressionMethod         (CM_PARSER parser);  // Добавить парсер нового метода в список поддерживаемых методов сжатия
int AddExternalCompressionMethod (CM_PARSER2 parser2, void *data);  // Добавить парсер внешнего метода сжатия с дополнительным параметром, который должен быть передан этому парсеру
#endif  // __cplusplus
void ClearExternalCompressorsTable (void);                          // Очистить таблицу внешних упаковщиков
#ifdef __cplusplus


// ****************************************************************************************************************************
// МЕТОД "СЖАТИЯ" STORING *****************************************************************************************************
// ****************************************************************************************************************************

// Реализация метода "сжатия" STORING
class STORING_METHOD : public COMPRESSION_METHOD
{
public:
  // Функции распаковки и упаковки
  virtual int decompress (CALLBACK_FUNC *callback, void *auxdata);
#ifndef FREEARC_DECOMPRESS_ONLY
  virtual int compress   (CALLBACK_FUNC *callback, void *auxdata);

  // Получить/установить объём памяти, используемой при упаковке/распаковке
  virtual MemSize GetCompressionMem        (void)               {return BUFFER_SIZE;}
  virtual void    SetCompressionMem        (MemSize)            {}
  virtual void    SetMinDecompressionMem   (MemSize)            {}
#endif
  virtual MemSize GetDecompressionMem      (void)               {return BUFFER_SIZE;}

  // Записать в buf[MAX_METHOD_STRLEN] строку, описывающую метод сжатия (функция, обратная к parse_STORING)
  virtual void ShowCompressionMethod (char *buf, bool purify)   {sprintf (buf, "storing");}
};

// Разборщик строки метода сжатия STORING
COMPRESSION_METHOD* parse_STORING (char** parameters);


// ****************************************************************************************************************************
// МЕТОД "СЖАТИЯ" CRC: читаем данные и ничего не пишем ************************************************************************
// ****************************************************************************************************************************

// Реализация метода "сжатия" crc
class CRC_METHOD : public COMPRESSION_METHOD
{
public:
  // Функции распаковки и упаковки
  virtual int decompress (CALLBACK_FUNC *callback, void *auxdata) {return FREEARC_ERRCODE_INTERNAL;}
#ifndef FREEARC_DECOMPRESS_ONLY
  virtual int compress   (CALLBACK_FUNC *callback, void *auxdata);

  // Получить/установить объём памяти, используемой при упаковке/распаковке
  virtual MemSize GetCompressionMem        (void)               {return BUFFER_SIZE;}
  virtual void    SetCompressionMem        (MemSize)            {}
  virtual void    SetMinDecompressionMem   (MemSize)            {}
#endif
  virtual MemSize GetDecompressionMem      (void)               {return BUFFER_SIZE;}

  // Записать в buf[MAX_METHOD_STRLEN] строку, описывающую метод сжатия (функция, обратная к parse_CRC)
  virtual void ShowCompressionMethod (char *buf, bool purify)   {sprintf (buf, "crc");}
};

// Разборщик строки метода "сжатия" crc
COMPRESSION_METHOD* parse_CRC (char** parameters);


// ****************************************************************************************************************************
// МЕТОД "СЖАТИЯ" FAKE: не читаем данные и ничего не пишем ********************************************************************
// ****************************************************************************************************************************

// Реализация метода "сжатия" fake
class FAKE_METHOD : public COMPRESSION_METHOD
{
public:
  // Функции распаковки и упаковки
  virtual int decompress (CALLBACK_FUNC *callback, void *auxdata) {return FREEARC_ERRCODE_INTERNAL;}
#ifndef FREEARC_DECOMPRESS_ONLY
  virtual int compress   (CALLBACK_FUNC *callback, void *auxdata) {return FREEARC_ERRCODE_INTERNAL;}

  // Получить/установить объём памяти, используемой при упаковке/распаковке
  virtual MemSize GetCompressionMem        (void)               {return BUFFER_SIZE;}
  virtual void    SetCompressionMem        (MemSize)            {}
  virtual void    SetMinDecompressionMem   (MemSize)            {}
#endif
  virtual MemSize GetDecompressionMem      (void)               {return BUFFER_SIZE;}

  // Записать в buf[MAX_METHOD_STRLEN] строку, описывающую метод сжатия (функция, обратная к parse_FAKE)
  virtual void ShowCompressionMethod (char *buf, bool purify)   {sprintf (buf, "fake");}
};

// Разборщик строки метода сжатия STORING
COMPRESSION_METHOD* parse_FAKE (char** parameters);

#endif  // __cplusplus


// ****************************************************************************************************************************
// (De)compress data from memory buffer (input) to another memory buffer (output)                                             *
// ****************************************************************************************************************************

// Структура, хранящая позицию в буферах чтения/записи при упаковке/распаковке в памяти
struct MemBuf
{
  MemBuf (void *input, int inputSize, void *output, int outputSize, CALLBACK_FUNC *_callback=0, void *_auxdata=0)
  {
    readPtr=(BYTE*)input, readLeft=inputSize, writePtr=(BYTE*)output, writeLeft=writeBufferSize=outputSize, callback=_callback, auxdata=_auxdata;
  }

  // Сколько данных было записано в буфер
  int written()  {return writeBufferSize-writeLeft;}

  BYTE *readPtr;          // текущая позиция читаемых данных (NULL, если надо читать данные через callback)
  int   readLeft;         // сколько байт ещё осталось во входном буфере
  BYTE *writePtr;         // текущая позиция записываемых данных (NULL, если надо записывать данные через callback)
  int   writeLeft;        // сколько байт ещё осталось в выходном буфере
  int   writeBufferSize;  // полный размер выходного буфера
  CALLBACK_FUNC *callback;
  void *auxdata;
};

// Callback-функция чтения/записи для (рас)паковки в памяти
int ReadWriteMem (const char *what, void *buf, int size, void *_membuf);


// ****************************************************************************************************************************
// ENCRYPTION ROUTINES *****************************************************************************************************
// ****************************************************************************************************************************

// Generates key based on password and salt using given number of hashing iterations
void Pbkdf2Hmac (const BYTE *pwd, int pwdSize, const BYTE *salt, int saltSize,
                 int numIterations, BYTE *key, int keySize);

int fortuna_size (void);


#ifdef __cplusplus
}       // extern "C"
#endif

#endif  // FREEARC_COMPRESSION_H
