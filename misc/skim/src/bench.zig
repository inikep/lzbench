const std = @import("std");
const Io = std.Io;
const zbench = @import("zbench");
const skim = @import("skim");

fn readFile(allocator: std.mem.Allocator, io: Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const buffer = try allocator.alloc(u8, @intCast(stat.size));

    _ = try file.readPositionalAll(io, buffer, 0);
    return buffer;
}

fn basename(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |idx| {
        return path[idx + 1 ..];
    }
    return path;
}

pub fn EncodeBenchmark(Encoder: type) type {
    return struct {
        const Self = @This();
        ctx: *Encoder,
        input: []const u8,
        output: []u8,

        pub fn init(ctx: *Encoder, input: []const u8, output: []u8) Self {
            return .{ .ctx = ctx, .input = input, .output = output };
        }

        pub fn run(self: *Self, _: std.mem.Allocator) void {
            _ = self.ctx.compressBlockToBuffer(self.input, self.output);
        }
    };
}

pub fn DecodeBenchmark(Decoder: type) type {
    return struct {
        const Self = @This();
        ctx: *Decoder,
        input: []const u8,
        output: []u8,

        pub fn init(ctx: *Decoder, input: []const u8, output: []u8) Self {
            return .{ .ctx = ctx, .input = input, .output = output };
        }

        pub fn run(self: *Self, _: std.mem.Allocator) void {
            _ = self.ctx.decompressBlockToBuffer(self.input, self.output);
        }
    };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;

    var bench = zbench.Benchmark.init(arena, .{});
    defer bench.deinit();

    if (args.len > 1) {
        const file_path = args[1];
        try writer.print("Loading {s}...\n", .{file_path});
        try writer.flush();

        const input_data = try readFile(arena, io, file_path);

        {
            const Encoder = skim.Encoder;
            const encoder = try arena.create(Encoder);
            encoder.* = try Encoder.init(arena);

            const output_data = try arena.alloc(u8, Encoder.outputBufferBound(input_data.len));
            const encode_name = try std.fmt.allocPrint(arena, "Encoder: {s}", .{basename(file_path)});

            const encode_param = try arena.create(EncodeBenchmark(Encoder));
            encode_param.* = EncodeBenchmark(Encoder).init(encoder, input_data, output_data);

            try bench.addParam(encode_name, @as(*const EncodeBenchmark(Encoder), encode_param), .{});

            const Decoder = skim.Decoder;
            const decoder = try arena.create(Decoder);
            decoder.* = try Decoder.init(arena);

            const compressed_buffer = try arena.alloc(u8, Encoder.outputBufferBound(input_data.len));
            const compressed_size = encoder.compressBlockToBuffer(input_data, compressed_buffer);
            const compressed_data = compressed_buffer[0..compressed_size];

            const decompressed_data = try arena.alloc(u8, input_data.len);
            const decode_name = try std.fmt.allocPrint(arena, "Decoder: {s}", .{basename(file_path)});

            const decode_param = try arena.create(DecodeBenchmark(Decoder));
            decode_param.* = DecodeBenchmark(Decoder).init(decoder, compressed_data, decompressed_data);

            try bench.addParam(decode_name, @as(*const DecodeBenchmark(Decoder), decode_param), .{});
        }
    }

    try writer.writeAll("\n");
    try writer.flush();
    try bench.run(io, std.Io.File.stdout());
}
