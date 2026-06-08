const std = @import("std");

pub const Encoder = encoder(u64, u8, u16);
pub const Decoder = decoder(u64, u8, u16);

fn LookupTable(comptime Key: type, comptime Value: type) type {
    return struct {
        const Self = @This();
        const SIZE = std.math.maxInt(Key) + 1;

        table: []Value,

        pub fn init(allocator: std.mem.Allocator) !Self {
            const table = try allocator.alloc(Value, SIZE);
            @memset(table, 0);
            return Self{ .table = table };
        }

        pub inline fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.table);
        }

        pub inline fn get(self: *const Self, key: Key) Value {
            return self.table[key];
        }

        pub inline fn set(self: *Self, key: Key, value: Value) void {
            self.table[key] = value;
        }
    };
}

fn NumberHasher(comptime Data: type, comptime Hash: type) type {
    return struct {
        const PRIME = switch (Data) {
            u128 => 0x9E3779B97F4A7C15F39CC0605CEDC7FD,
            u64 => 0x9E3779B97F4A7C15,
            u32 => 0x9D6EF916,
            u16 => 0x9E3B,
            u8 => 0x9D,
            else => @compileError("Unsupported Data type size for Hasher"),
        };
        const SHIFT = @bitSizeOf(Data) - @bitSizeOf(Hash);

        pub inline fn hash(data: Data) Hash {
            return @truncate((data *% PRIME) >> SHIFT);
        }
    };
}

pub fn encoder(comptime Word: type, comptime Header: type, comptime Hash: type) type {
    const Hasher = NumberHasher(Word, Hash);
    const Table = LookupTable(Hash, Word);
    const Size = u64;

    const HEADER_BITS = @bitSizeOf(Header);
    const WORD_BYTES = @sizeOf(Word);
    const HEADER_BYTES = @sizeOf(Header);
    const HASH_BYTES = @sizeOf(Hash);
    const SIZE_BYTES = @sizeOf(Size);
    const BATCH_BYTES = HEADER_BITS * WORD_BYTES;

    return struct {
        const Self = @This();
        table: Table,

        pub fn init(allocator: std.mem.Allocator) !Self {
            return .{ .table = try Table.init(allocator) };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.table.deinit(allocator);
        }

        pub inline fn outputBufferBound(len: usize) usize {
            const blocks = len / BATCH_BYTES;
            return len + (blocks * HEADER_BYTES) + HEADER_BYTES + WORD_BYTES + SIZE_BYTES;
        }

        pub fn compressBlockToBuffer(self: *Self, input: []const u8, output: []u8) usize {
            @setRuntimeSafety(false);

            var input_index: usize = 0;
            var output_index: usize = 0;
            const loop_limit = (input.len / BATCH_BYTES) * BATCH_BYTES;

            std.mem.writeInt(Size, output[0..SIZE_BYTES], @intCast(input.len), .little);
            output_index += SIZE_BYTES;

            while (input_index < loop_limit) {
                const header_pos = output_index;
                output_index += HEADER_BYTES;
                var header: Header = 0;

                inline for (0..HEADER_BITS) |token_index| {
                    const word = std.mem.readInt(Word, input[input_index..][0..WORD_BYTES], .little);
                    input_index += WORD_BYTES;

                    const hash = Hasher.hash(word);
                    if (word == self.table.get(hash)) {
                        std.mem.writeInt(Hash, output[output_index..][0..HASH_BYTES], hash, .little);
                        output_index += HASH_BYTES;
                        header |= 1 << token_index;
                    } else {
                        std.mem.writeInt(Word, output[output_index..][0..WORD_BYTES], word, .little);
                        output_index += WORD_BYTES;
                        self.table.set(hash, word);
                    }
                }

                std.mem.writeInt(Header, output[header_pos..][0..HEADER_BYTES], header, .little);
            }

            const remaining = input.len - input_index;
            if (remaining != 0) {
                @memcpy(output[output_index .. output_index + remaining], input[input_index .. input_index + remaining]);
                output_index += remaining;
            }

            return output_index;
        }
    };
}

pub fn decoder(comptime Word: type, comptime Header: type, comptime Hash: type) type {
    const Hasher = NumberHasher(Word, Hash);
    const Table = LookupTable(Hash, Word);
    const Size = u64;

    const HEADER_BITS = @bitSizeOf(Header);
    const WORD_BYTES = @sizeOf(Word);
    const HEADER_BYTES = @sizeOf(Header);
    const HASH_BYTES = @sizeOf(Hash);
    const SIZE_BYTES = @sizeOf(Size);
    const BATCH_BYTES = HEADER_BITS * WORD_BYTES;

    return struct {
        const Self = @This();
        table: Table,

        pub fn init(allocator: std.mem.Allocator) !Self {
            return .{ .table = try Table.init(allocator) };
        }

        pub inline fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.table.deinit(allocator);
        }

        pub inline fn exactOutputLength(input: []const u8) usize {
            return @intCast(std.mem.readInt(Size, input[0..SIZE_BYTES], .little));
        }

        pub fn decompressBlockToBuffer(self: *Self, input: []const u8, output: []u8) usize {
            @setRuntimeSafety(false);

            const len = exactOutputLength(input);
            const loop_limit = (len / BATCH_BYTES) * BATCH_BYTES;

            var input_index: usize = SIZE_BYTES;
            var output_index: usize = 0;

            while (output_index < loop_limit) {
                const header = std.mem.readInt(Header, input[input_index..][0..HEADER_BYTES], .little);
                input_index += HEADER_BYTES;

                inline for (0..HEADER_BITS) |token_index| {
                    var word: Word = undefined;

                    if ((header & (1 << token_index)) != 0) {
                        const hash = std.mem.readInt(Hash, input[input_index..][0..HASH_BYTES], .little);
                        input_index += HASH_BYTES;
                        word = self.table.get(hash);
                    } else {
                        word = std.mem.readInt(Word, input[input_index..][0..WORD_BYTES], .little);
                        input_index += WORD_BYTES;
                        self.table.set(Hasher.hash(word), word);
                    }

                    std.mem.writeInt(Word, output[output_index..][0..WORD_BYTES], word, .little);
                    output_index += WORD_BYTES;
                }
            }

            const remaining = len - output_index;
            if (remaining != 0) {
                @memcpy(output[output_index .. output_index + remaining], input[input_index .. input_index + remaining]);
                input_index += remaining;
            }

            return input_index;
        }
    };
}

test LookupTable {
    const Table = LookupTable(u8, u16);
    var table = try Table.init(std.testing.allocator);
    defer table.deinit(std.testing.allocator);

    try std.testing.expectEqual(0, table.get(0));
    table.set(0, 1);
    try std.testing.expectEqual(1, table.get(0));
}

test NumberHasher {
    try std.testing.expectEqual(157, NumberHasher(u32, u8).hash(1));
}

test "Encoder / Decoder full cycle" {
    var compressor = try Encoder.init(std.testing.allocator);
    defer compressor.deinit(std.testing.allocator);

    var decompressor = try Decoder.init(std.testing.allocator);
    defer decompressor.deinit(std.testing.allocator);

    const input_data = "A" ** 64 ++ "B" ** 64 ++ "C" ** 17;

    var compressed_buffer: [Encoder.outputBufferBound(input_data.len)]u8 = undefined;
    var decompressed_buffer: [input_data.len]u8 = undefined;

    const compressed_size = compressor.compressBlockToBuffer(input_data, &compressed_buffer);
    _ = decompressor.decompressBlockToBuffer(compressed_buffer[0..compressed_size], &decompressed_buffer);

    try std.testing.expectEqual(input_data.len, input_data.len);
    try std.testing.expectEqualStrings(input_data, &decompressed_buffer);
}

const c_allocator = std.heap.c_allocator;

export fn skim_encoder_create() ?*Encoder {
    const enc = c_allocator.create(Encoder) catch return null;
    errdefer c_allocator.destroy(enc);
    enc.* = Encoder.init(c_allocator) catch return null;
    return enc;
}

export fn skim_encoder_destroy(encoder_ptr: ?*Encoder) void {
    if (encoder_ptr) |enc| {
        enc.deinit(c_allocator);
        c_allocator.destroy(enc);
    }
}

export fn skim_encoder_output_buffer_bound(len: usize) usize {
    return Encoder.outputBufferBound(len);
}

export fn skim_encoder_compress(
    encoder_ptr: ?*Encoder,
    input_ptr: ?[*]const u8,
    input_len: usize,
    output_ptr: ?[*]u8,
) usize {
    const enc = encoder_ptr orelse return 0;
    const in_ptr = input_ptr orelse return 0;
    const out_ptr = output_ptr orelse return 0;

    if (input_len == 0) return 0;

    const input = in_ptr[0..input_len];

    const output_bound = Encoder.outputBufferBound(input_len);
    const output = out_ptr[0..output_bound];

    return enc.compressBlockToBuffer(input, output);
}

export fn skim_encoder_reset(encoder_ptr: ?*Encoder) void {
    if (encoder_ptr) |enc| {
        @memset(enc.table.table, 0);
    }
}

export fn skim_decoder_create() ?*Decoder {
    const dec = c_allocator.create(Decoder) catch return null;
    errdefer c_allocator.destroy(dec);
    dec.* = Decoder.init(c_allocator) catch return null;
    return dec;
}

export fn skim_decoder_destroy(decoder_ptr: ?*Decoder) void {
    if (decoder_ptr) |dec| {
        dec.deinit(c_allocator);
        c_allocator.destroy(dec);
    }
}

export fn skim_decoder_exact_output_length(input_ptr: ?[*]const u8, input_len: usize) usize {
    const in_ptr = input_ptr orelse return 0;
    const input = in_ptr[0..input_len];
    return Decoder.exactOutputLength(input);
}

export fn skim_decoder_decompress(
    decoder_ptr: ?*Decoder,
    input_ptr: ?[*]const u8,
    input_len: usize,
    output_ptr: ?[*]u8,
    output_len: usize,
) usize {
    const dec = decoder_ptr orelse return 0;
    const in_ptr = input_ptr orelse return 0;
    const out_ptr = output_ptr orelse return 0;

    if (input_len == 0 or output_len == 0) return 0;

    const input = in_ptr[0..input_len];
    const output = out_ptr[0..output_len];

    return dec.decompressBlockToBuffer(input, output);
}

export fn skim_decoder_reset(decoder_ptr: ?*Decoder) void {
    if (decoder_ptr) |dec| {
        @memset(dec.table.table, 0);
    }
}
