const std = @import("std");
const skim = @import("skim");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const io = init.io;

    const prog = if (args.len > 0) args[0] else "skim";
    const valid_args = args.len == 4 and (std.mem.eql(u8, args[1], "-c") or std.mem.eql(u8, args[1], "-d")) and !std.mem.eql(u8, args[2], args[3]) or std.mem.eql(u8, args[2], "-") and std.mem.eql(u8, args[3], "-");

    if (!valid_args) {
        std.debug.print("Usage: {s} [-c | -d] <input> <output>\n", .{prog});
        return error.InvalidArguments;
    }

    const is_decode = args[1][1] == 'd';
    const cwd = std.Io.Dir.cwd();

    const input_file = if (std.mem.eql(u8, args[2], "-")) std.Io.File.stdin() else try cwd.openFile(io, args[2], .{});
    defer input_file.close(io);

    const output_file = if (std.mem.eql(u8, args[3], "-")) std.Io.File.stdout() else try cwd.createFile(io, args[3], .{});
    defer output_file.close(io);

    var reader_wrap = input_file.readerStreaming(io, &.{});
    const reader = &reader_wrap.interface;

    const input = try reader.allocRemaining(arena, .unlimited);
    if (input.len == 0) return error.EmptyInput;

    var writer_wrap = output_file.writerStreaming(io, &.{});
    const writer = &writer_wrap.interface;

    const BLOCK_SIZE = comptime std.math.maxInt(u21);
    var offset: usize = 0;

    if (is_decode) {
        var decoder = try skim.Decoder.init(arena);
        defer decoder.deinit(arena);
        const buffer = try arena.alloc(u8, BLOCK_SIZE);

        while (offset < input.len) {
            const chunk = input[offset..];
            const out_len = skim.Decoder.exactOutputLength(chunk);

            const consumed = decoder.decompressBlockToBuffer(chunk, buffer[0..out_len]);
            try writer.writeAll(buffer[0..out_len]);

            offset += consumed;
        }
    } else {
        var encoder = try skim.Encoder.init(arena);
        defer encoder.deinit(arena);
        const buffer = try arena.alloc(u8, comptime skim.Encoder.outputBufferBound(BLOCK_SIZE));

        while (offset < input.len) {
            const chunk_size = @min(input.len - offset, BLOCK_SIZE);
            const chunk = input[offset .. offset + chunk_size];

            const len = encoder.compressBlockToBuffer(chunk, buffer);
            try writer.writeAll(buffer[0..len]);

            offset += chunk_size;
        }
    }
}
