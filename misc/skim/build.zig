const std = @import("std");

pub fn build(b: *std.Build) void {
    // Options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Packages
    const zbench_pkg = b.dependency("zbench", .{ .target = target, .optimize = optimize });

    // Modules
    const zbench_mod = zbench_pkg.module("zbench");

    const root_mod = b.addModule("skim", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "skim", .module = root_mod },
        },
    });

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "skim", .module = root_mod }, .{ .name = "zbench", .module = zbench_mod } },
    });

    // Libraries
    const root_lib = b.addLibrary(.{
        .name = "skim",
        .linkage = .dynamic,
        .root_module = root_mod,
        .use_llvm = true,
    });

    // Directories
    const docs_dir = b.addInstallDirectory(.{
        .source_dir = root_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });

    // Binaries
    const main_bin = b.addExecutable(.{
        .name = "skim",
        .root_module = main_mod,
        .use_llvm = true,
    });

    const bench_bin = b.addExecutable(.{
        .name = "benchmarks",
        .root_module = bench_mod,
        .use_llvm = true,
    });

    const root_test_bin = b.addTest(.{
        .name = "root_tests",
        .root_module = root_mod,
        .use_llvm = true,
    });

    const main_test_bin = b.addTest(.{
        .name = "main_tests",
        .root_module = main_mod,
        .use_llvm = true,
    });

    // Commands
    const run_cmd = b.addRunArtifact(main_bin);
    const bench_cmd = b.addRunArtifact(bench_bin);
    const test_root_cmd = b.addRunArtifact(root_test_bin);
    const test_main_cmd = b.addRunArtifact(main_test_bin);

    if (b.args) |args| {
        run_cmd.addArgs(args);
        bench_cmd.addArgs(args);
    }

    run_cmd.step.dependOn(b.getInstallStep());

    // Steps - Run
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Steps - Benchmarks
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench_cmd.step);

    // Steps - Tests
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&test_main_cmd.step);
    test_step.dependOn(&test_root_cmd.step);

    // Steps - Docs
    const docs_step = b.step("docs", "Install docs into zig-out/docs");
    docs_step.dependOn(&docs_dir.step);

    // Install
    b.installArtifact(main_bin);
    b.installArtifact(root_test_bin);
    b.installArtifact(root_lib);
}
