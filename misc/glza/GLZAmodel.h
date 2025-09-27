enum { TOP = 1 << 24, BUF_SIZE = 0x40000 };
enum { UP_FREQ_SYM_TYPE = 1, FREQ_SYM_TYPE_BOT1 = 0x4000, FREQ_SYM_TYPE_BOT2 = 0x2000, FREQ_SYM_TYPE_BOT3 = 0x1000 };
enum { UP_FREQ_MTF_POS = 4, FREQ_MTF_POS_BOT = 0x2000 };
enum { UP_FREQ_SID = 3, FREQ_SID_BOT = 0x1000 };
enum { UP_FREQ_INST = 8, FREQ_INST_BOT = 0x8000 };
enum { FREQ_ERG_BOT = 0x2000 };
enum { FREQ_GO_MTF_BOT = 0x2000 };
enum { FREQ_WORD_TAG_BOT = 0x1000 };
enum { UP_FREQ_FIRST_CHAR = 8, FREQ_FIRST_CHAR_BOT = 0x2000 };
enum { NOT_CAP = 0, CAP = 1 };
enum { LEVEL0 = 0, LEVEL0_CAP = 1, LEVEL1 = 2, LEVEL1_CAP = 3 };

#define START_UTF8_2BYTE_SYMBOLS 0x80
#define START_UTF8_3BYTE_SYMBOLS 0x800
#define START_UTF8_4BYTE_SYMBOLS 0x10000
#define MAX_INSTANCES_FOR_REMOVE 15

uint32_t ReadLow();
uint32_t ReadRange();
void NormalizeEncoder(uint32_t bot);
void NormalizeDecoder(uint32_t bot);
void InitFirstCharBin(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length, uint8_t cap_symbol_defined,
    uint8_t cap_lock_symbol_defined);
void InitFirstCharBinBinary(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length);
void InitTrailingCharBin(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length);
void InitTrailingCharBinary(uint8_t trailing_char, uint8_t * symbol_lengths);
void InitBaseSymbolCap(uint8_t BaseSymbol, uint8_t new_symbol_code_length, uint8_t * cap_symbol_defined_ptr,
    uint8_t * cap_lock_symbol_defined_ptr, uint8_t * symbol_lengths);
void UpFreqMtfQueueNum(uint8_t Context, uint8_t mtf_queue_number);
void IncreaseRange(uint32_t low_ranges, uint32_t ranges);
void DoubleRange();
void DoubleRangeDown();
void SetOutBuffer(uint8_t * bufptr);
void WriteOutBuffer(uint8_t value);
void EncodeDictType1(uint8_t Context1);
void EncodeDictType2(uint8_t Context1, uint8_t Context2);
void EncodeDictType3(uint8_t Context1, uint8_t Context2, uint8_t Context3);
void EncodeNewType1(uint8_t Context1);
void EncodeNewType2(uint8_t Context1, uint8_t Context2);
void EncodeNewType3(uint8_t Context1, uint8_t Context2, uint8_t Context3);
void EncodeMtfType1(uint8_t Context1);
void EncodeMtfType2(uint8_t Context1, uint8_t Context2);
void EncodeMtfType3(uint8_t Context1, uint8_t Context2, uint8_t Context3);
void EncodeMtfFirst(uint8_t Context, uint8_t First);
void EncodeMtfPos(uint8_t position, uint16_t QueueSize);
void EncodeMtfPosAz(uint8_t position, uint16_t QueueSize);
void EncodeMtfPosSpace(uint8_t position, uint16_t QueueSize);
void EncodeMtfPosOther(uint8_t position, uint16_t QueueSize);
void EncodeSID(uint8_t Context, uint8_t SIDSymbol);
void EncodeExtraLength(uint8_t Symbol);
void EncodeINST(uint8_t Context, uint8_t SIDSymbol, uint8_t Symbol);
void EncodeERG(uint16_t Context1, uint16_t Context2, uint8_t Symbol);
void EncodeGoMtf(uint16_t Context1, uint8_t Context2, uint8_t Symbol);
void EncodeWordTag(uint8_t Symbol, uint8_t Context);
void EncodeShortDictionarySymbol(uint16_t BinNum, uint16_t DictionaryBins, uint16_t CodeBins);
void EncodeLongDictionarySymbol(uint32_t BinCode, uint16_t BinNum, uint16_t DictionaryBins, uint8_t CodeLength,
    uint16_t CodeBins);
void EncodeBaseSymbol(uint32_t BaseSymbol, uint32_t NumBaseSymbols, uint32_t NormBaseSymbols);
void EncodeFirstChar(uint8_t Symbol, uint8_t SymType, uint8_t LastChar);
void EncodeFirstCharBinary(uint8_t Symbol, uint8_t LastChar);
void UpdateFirstChar(uint8_t Symbol, uint8_t SymType, uint8_t LastChar);
void WriteInCharNum(uint32_t value);
uint32_t ReadOutCharNum();
void InitEncoder(uint8_t max_regular_code_length, uint8_t max_base_code, uint8_t num_inst_codes, uint8_t cap_encoded,
    uint8_t UTF8_compliant, uint8_t use_mtf);
void FinishEncoder();
uint8_t DecodeSymType1(uint8_t Context);
uint8_t DecodeSymType2(uint8_t Context, uint8_t Context2);
uint8_t DecodeSymType3(uint8_t Context, uint8_t Context2, uint8_t Context3);
uint8_t DecodeMtfFirst(uint8_t Context);
uint8_t DecodeMtfPos(uint16_t QueueSize);
uint8_t DecodeMtfPosAz(uint16_t QueueSize);
uint8_t DecodeMtfPosSpace(uint16_t QueueSize);
uint8_t DecodeMtfPosOther(uint16_t QueueSize);
uint8_t DecodeSID(uint8_t Context);
uint8_t DecodeExtraLength();
uint8_t DecodeINST(uint8_t Context, uint8_t SIDSymbol);
uint8_t DecodeERG(uint16_t Context1, uint16_t Context2);
uint8_t DecodeGoMtf(uint16_t Context1, uint8_t Context2);
uint8_t DecodeWordTag(uint8_t Context);
uint16_t DecodeBin(uint16_t DictionaryBins);
uint32_t DecodeBinCode(uint8_t Bits);
uint32_t DecodeBaseSymbol(uint32_t NumBaseSymbols);
uint32_t DecodeBaseSymbolCap(uint32_t NumBaseSymbols);
uint8_t DecodeFirstChar(uint8_t SymType, uint8_t LastChar);
uint8_t DecodeFirstCharBinary(uint8_t LastChar);
void InitDecoder(uint8_t max_regular_code_length, uint8_t max_base_code, uint8_t num_inst_codes, uint8_t cap_encoded,
    uint8_t UTF8_compliant, uint8_t use_mtf, uint8_t * inbuf);
