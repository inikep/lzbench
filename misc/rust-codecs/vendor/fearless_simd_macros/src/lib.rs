// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![doc = include_str!("../README.md")]

use core::mem;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::fold::{self, Fold};
use syn::{Attribute, FnArg, ItemFn, Pat, Result};

/// Run a SIMD-generic function body with the token's target features enabled.
///
/// The first typed parameter after an optional `self` receiver is used as the
/// SIMD token carrier: a token, vector, mask, or a value implementing
/// `fearless_simd::ExtractToken`. The library must be in scope as `fearless_simd`;
/// for a renamed dependency, import it with `use simd_backend as fearless_simd;` in the
/// containing module. See the [crate-level documentation](crate) for the complete
/// expansion, supported function forms, and semantic caveats.
#[proc_macro_attribute]
pub fn simd(args: TokenStream, item: TokenStream) -> TokenStream {
    expand(args.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(args: TokenStream2, item: TokenStream2) -> Result<TokenStream2> {
    if !args.is_empty() {
        return Err(syn::Error::new_spanned(
            args,
            "`#[simd]` does not accept arguments",
        ));
    }

    // ItemFn's signature grammar also accepts body-bearing inherent and trait
    // methods. Parsing the item rejects bodyless and specialization methods.
    let mut function = syn::parse2::<ItemFn>(item).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!("`#[simd]` can only be used on function and method definitions: {error}"),
        )
    })?;

    function.modifiers.require_empty()?;
    reject_unsupported_signature(&function)?;
    reject_unsupported_attributes(&function.attrs)?;

    let original_carrier = validate_token_carrier(&function)?;
    // Token extraction uses the outer binding. Also mark the original binding
    // in the body closure as used, so unused carriers do not trigger warnings.
    let use_carrier = original_carrier.map(|carrier| quote!(let _ = #carrier;));
    let original_statements = mem::take(&mut function.block.stmts);
    // Give the closure the same expected return type so branch and early-return
    // coercions happen inside its body. Closures cannot name `impl Trait`, so
    // infer those parts while preserving the surrounding type structure.
    let closure_output = InferImplTrait.fold_return_type(function.sig.output.clone());

    // Wrapping the body in simd.vectorize(|| ...) puts its captured arguments
    // into a closure struct. If the target-feature helper remains out of line,
    // that struct can be passed through memory, spilling arguments that would
    // otherwise fit in registers. Instead, make the body a FnOnce(A0, A1, ...)
    // and pass each argument separately through the dispatcher. This preserves
    // register passing without forcing large bodies to inline into every caller.
    //
    // Preserve outer parameter names for documentation and IDEs when a pattern
    // names the whole argument. Otherwise, give it a fresh name so we can forward
    // its whole value. Move the original patterns into the closure's parameters
    // so the body keeps its original bindings and borrowing behavior.
    // Helper parameter names must always be fresh: caller names such as `entry`
    // could otherwise collide with items inside the dispatch helpers.
    // Fresh generic argument types let the helpers forward these values without
    // having to reproduce the outer function's generics, lifetimes, or Self.
    // Keeping one closure body also preserves a single opaque return type when
    // the function returns impl Trait.
    let mut parameters = Vec::new();
    let mut arguments = Vec::new();
    let mut helper_arguments = Vec::new();
    let mut argument_types = Vec::new();
    for (index, argument) in function.sig.inputs.iter_mut().enumerate() {
        let FnArg::Typed(argument) = argument else {
            // Keep `self` captured so its uses, including inside nested macros,
            // retain their original meaning without rewriting the body.
            continue;
        };
        if has_conditional_attributes(&argument.attrs) {
            // FnOnce's argument types cannot carry cfg attributes. Preserve
            // conditional parameters as captures, just as in the old expansion.
            continue;
        }
        let helper_name = Ident::new(&format!("__fearless_argument_{index}"), Span::mixed_site());
        let name = if let Some(name) = argument_binding(&argument.pat) {
            let name = name.clone();
            // A bare identifier could resolve to a constant or unit constructor
            // instead of binding the argument. Require a binding so forwarding
            // cannot construct a new value; explicit path patterns use the fresh
            // name fallback instead. Keep this check out of the visible signature.
            function.block.stmts.push(syn::parse_quote! {
                #[allow(
                    clippy::redundant_pattern,
                    reason = "force an identifier binding instead of a constant or unit constructor"
                )]
                let #name @ _ = #name;
            });
            name
        } else {
            helper_name.clone()
        };
        let ty = Ident::new(&format!("__FearlessArgument{index}"), Span::mixed_site());
        let pattern = mem::replace(&mut argument.pat, Box::new(syn::parse_quote!(#name)));
        // Move lint attributes with the original binding. Duplicating `expect`
        // on the now-used outer parameter would leave an unfulfilled expectation.
        let attrs = mem::take(&mut argument.attrs);
        parameters.push(quote!(#(#attrs)* #pattern));
        arguments.push(name);
        helper_arguments.push(helper_name);
        argument_types.push(ty);
    }
    // Borrow the first argument only long enough to extract its token, before
    // forwarding the original value. Carriers need not be Copy. Use the trait
    // explicitly so an inherent token() method cannot change dispatch.
    let carrier = &arguments[0];
    let token = quote! {
        fearless_simd::ExtractToken::token(&#carrier)
    };

    // Inner function attributes are held in function.attrs by Syn. Leaving
    // them there keeps them at the beginning of the outer function body,
    // rather than changing their scope by moving them into this closure.
    // Keep generated wrapper tokens on their normal macro-expansion spans.
    // Giving the entire call the token parameter's source span makes Clippy's
    // `semicolon_if_nothing_returned` lint fire on unit-returning functions.
    // Keep the closure directly in the call, after the arguments: the FnOnce
    // bound then infers its parameter types and their borrowed-return lifetimes.
    // The library macro owns the unsafe calls and resolves proof types through
    // $crate. A lookalike `fearless_simd` module cannot spoof those proofs.
    let dispatch_call: syn::Expr = syn::parse_quote! {
        (fearless_simd::__fearless_simd_dispatch!(#(#argument_types => #helper_arguments),*)).call(
            #token, #(#arguments,)*
            #[inline(always)]
            |#(#parameters),*| #closure_output { #use_carrier #(#original_statements)* }
        )
    };
    function
        .block
        .stmts
        .push(syn::Stmt::Expr(dispatch_call, None));

    Ok(quote!(#function))
}

fn argument_binding(pattern: &Pat) -> Option<&Ident> {
    match pattern {
        Pat::Ident(pattern) => Some(&pattern.ident),
        Pat::Paren(pattern) => argument_binding(&pattern.pat),
        _ => None,
    }
}

struct InferImplTrait;

impl Fold for InferImplTrait {
    fn fold_type(&mut self, ty: syn::Type) -> syn::Type {
        match ty {
            syn::Type::ImplTrait(_) => syn::parse_quote!(_),
            ty => fold::fold_type(self, ty),
        }
    }
}

fn reject_unsupported_signature(function: &ItemFn) -> Result<()> {
    if let Some(asyncness) = &function.sig.asyncness {
        return Err(syn::Error::new(
            asyncness.span,
            "`#[simd]` does not support async functions",
        ));
    }
    if let Some(constness) = &function.sig.constness {
        return Err(syn::Error::new(
            constness.span,
            "`#[simd]` does not support const functions",
        ));
    }
    if let Some(variadic) = &function.sig.variadic {
        return Err(syn::Error::new_spanned(
            variadic,
            "`#[simd]` does not support variadic functions",
        ));
    }
    Ok(())
}

fn reject_unsupported_attributes(attrs: &[Attribute]) -> Result<()> {
    for attr in attrs {
        let reason = if is_attribute(attr, "track_caller") {
            Some("`#[simd]` cannot preserve `#[track_caller]` through its closure")
        } else if is_attribute(attr, "naked") {
            Some("`#[simd]` cannot be used on a naked function")
        } else if is_attribute(attr, "instruction_set") {
            Some("`#[simd]` cannot be combined with `#[instruction_set]`")
        } else {
            None
        };

        if let Some(reason) = reason {
            return Err(syn::Error::new_spanned(attr, reason));
        }
    }
    Ok(())
}

fn is_attribute(attr: &Attribute, name: &str) -> bool {
    if attr.path().is_ident(name) {
        return true;
    }

    // Attributes with safety obligations use `#[unsafe(attribute)]` syntax.
    // Naked functions require this form on supported Rust releases.
    attr.path().is_ident("unsafe")
        && attr
            .parse_args::<syn::Path>()
            .is_ok_and(|path| path.is_ident(name))
}

fn validate_token_carrier(function: &ItemFn) -> Result<Option<Ident>> {
    let Some(argument) = function
        .sig
        .inputs
        .iter()
        .find_map(|argument| match argument {
            FnArg::Receiver(_) => None,
            FnArg::Typed(argument) => Some(argument),
        })
    else {
        return Err(syn::Error::new_spanned(
            &function.sig.inputs,
            "`#[simd]` requires a SIMD token carrier parameter after any receiver",
        ));
    };

    reject_conditional_attributes(&argument.attrs)?;

    match &*argument.pat {
        Pat::Ident(pattern) => {
            reject_conditional_attributes(&pattern.attrs)?;
            if let Some((at, _)) = &pattern.subpat {
                return Err(syn::Error::new(
                    at.span,
                    "the SIMD token carrier parameter cannot use an `@` subpattern",
                ));
            }
            Ok(Some(pattern.ident.clone()))
        }
        Pat::Wild(pattern) => {
            reject_conditional_attributes(&pattern.attrs)?;
            Ok(None)
        }
        pattern => Err(syn::Error::new_spanned(
            pattern,
            "the SIMD token carrier parameter must be an identifier or `_`",
        )),
    }
}

fn reject_conditional_attributes(attrs: &[Attribute]) -> Result<()> {
    if let Some(attr) = attrs
        .iter()
        .find(|attr| attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr"))
    {
        return Err(syn::Error::new_spanned(
            attr,
            "the SIMD token carrier parameter cannot be conditional",
        ));
    }
    Ok(())
}

fn has_conditional_attributes(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|attr| attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr"))
}

#[cfg(test)]
mod tests {
    use quote::{ToTokens, quote};
    use syn::{AttrStyle, Expr, ItemFn, Stmt};

    use super::expand;

    fn expand_ok(item: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        expand(proc_macro2::TokenStream::new(), item).expect("macro expansion should succeed")
    }

    fn expand_err(item: proc_macro2::TokenStream) -> String {
        expand(proc_macro2::TokenStream::new(), item)
            .expect_err("macro expansion should fail")
            .to_string()
    }

    #[test]
    fn unit_returns_remain_tail_expressions() {
        for item in [
            quote! {
                fn implicit_unit<S: Simd>(simd: S) { let _ = simd.level(); }
            },
            quote! {
                fn explicit_unit<S: Simd>(simd: S) -> () { let _ = simd.level(); }
            },
        ] {
            let expanded = expand_ok(item);
            let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");
            let Some(Stmt::Expr(Expr::MethodCall(call), None)) = parsed.block.stmts.last() else {
                panic!("unit function tail should be a method call");
            };
            let Some(Expr::Closure(closure)) = call.args.last() else {
                panic!("last dispatcher argument should be a closure");
            };

            assert_eq!(
                closure.output.to_token_stream().to_string(),
                parsed.sig.output.to_token_stream().to_string()
            );
        }
    }

    #[test]
    fn non_unit_return_remains_a_tail_expression() {
        let expanded = expand_ok(quote! {
            fn non_unit<S: Simd>(simd: S) -> u32 { 42 }
        });
        let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");

        assert!(matches!(
            parsed.block.stmts.last(),
            Some(Stmt::Expr(Expr::MethodCall(_), None))
        ));
    }

    #[test]
    fn preserves_signature_attributes_and_inner_attributes() {
        let expanded = expand_ok(quote! {
            #[doc = "docs"]
            #[inline(never)]
            #[target_feature(enable = "sse2")]
            unsafe extern "C" fn operation<'a, S, T>(simd: S, value: &'a T) -> &'a T
            where
                S: Simd,
            {
                #![allow(unused_unsafe)]
                unsafe { value }
            }
        });
        let parsed: ItemFn = syn::parse2(expanded.clone()).expect("expanded function parses");

        assert!(matches!(parsed.sig.safety, syn::Safety::Unsafe(_)));
        assert!(parsed.sig.abi.is_some());
        assert!(parsed.sig.generics.where_clause.is_some());
        assert_eq!(
            parsed
                .attrs
                .iter()
                .filter(|attr| matches!(attr.style, AttrStyle::Inner(_)))
                .count(),
            1
        );

        let text = expanded.to_string();
        let inner_attr = text
            .find("# ! [allow")
            .expect("inner attribute is retained");
        let call = text
            .find("__fearless_simd_dispatch")
            .expect("dispatcher invocation exists");
        assert!(inner_attr < call);
        assert_eq!(text.matches("inline (never)").count(), 1);
        assert_eq!(text.matches("inline (always)").count(), 1);
        assert!(text.contains("target_feature"));
    }

    #[test]
    fn selects_first_typed_parameter_after_receiver() {
        let expanded = expand_ok(quote! {
            fn method<S: Simd>(&self, mut backend: S, value: u32) -> u32 {
                backend.level();
                value
            }
        });
        let text = expanded.to_string();

        assert!(text.contains("backend : S"));
        assert!(text.contains("| mut backend , value |"));
    }

    #[test]
    fn gives_a_wildcard_token_a_private_binding() {
        let expanded = expand_ok(quote! {
            fn operation<S: Simd>(_: S, value: u32) -> u32 { value }
        });
        let text = expanded.to_string();

        assert_eq!(text.matches("__fearless_simd_token").count(), 0);
        assert!(text.contains("__fearless_argument_0 : S"));
        assert!(text.contains("| _ , value |"));
    }

    #[test]
    fn wildcard_binding_does_not_rename_a_user_binding_with_the_same_spelling() {
        let expanded = expand_ok(quote! {
            fn operation<S: Simd>(_: S, __fearless_simd_token: u32) -> u32 {
                __fearless_simd_token
            }
        });
        let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");

        assert_eq!(parsed.sig.inputs.len(), 2);
        assert_eq!(
            parsed.sig.inputs.to_token_stream().to_string(),
            quote!(__fearless_argument_0: S, __fearless_simd_token: u32).to_string()
        );
    }

    #[test]
    fn preserves_named_parameters_and_original_closure_patterns() {
        let expanded = expand_ok(quote! {
            fn operation<S: Simd>(
                simd: S,
                value: String,
                mut mutable: String,
                ref borrowed: String,
                ref mut borrowed_mut: String,
                r#type: u32,
                whole @ (left, right): (u32, u32),
                (parenthesized): String,
            ) {}
        });
        let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");

        assert_eq!(
            parsed.sig.inputs.to_token_stream().to_string(),
            quote!(
                simd: S,
                value: String,
                mutable: String,
                borrowed: String,
                borrowed_mut: String,
                r#type: u32,
                whole: (u32, u32),
                parenthesized: String,
            )
            .to_string()
        );
        let Some(Stmt::Expr(Expr::MethodCall(call), None)) = parsed.block.stmts.last() else {
            panic!("function tail should be a method call");
        };
        let Some(Expr::Closure(closure)) = call.args.last() else {
            panic!("last dispatcher argument should be a closure");
        };
        assert_eq!(
            closure.inputs.to_token_stream().to_string(),
            quote!(
                simd, value, mut mutable, ref borrowed, ref mut borrowed_mut,
                r#type, whole @ (left, right), (parenthesized)
            )
            .to_string()
        );
    }

    #[test]
    fn accepts_default_trait_method_syntax() {
        let expanded = expand_ok(quote! {
            fn operation<S: Simd>(&self, simd: S) -> u32 { 42 }
        });
        assert!(expanded.to_string().contains("| simd |"));
    }

    #[test]
    fn rejects_attribute_arguments() {
        let error = expand(
            quote!(token = simd),
            quote!(
                fn f<S: Simd>(simd: S) {}
            ),
        )
        .expect_err("arguments should be rejected")
        .to_string();
        assert_eq!(error, "`#[simd]` does not accept arguments");
    }

    #[test]
    fn rejects_unsupported_signatures() {
        assert!(
            expand_err(quote!(
                async fn f<S: Simd>(simd: S) {}
            ))
            .contains("async functions")
        );
        assert!(
            expand_err(quote!(
                const fn f<S: Simd>(simd: S) {}
            ))
            .contains("const functions")
        );
        assert!(
            expand_err(quote!(
                unsafe extern "C" fn f<S: Simd>(simd: S, ...) {}
            ))
            .contains("variadic functions")
        );
        assert!(expand(quote!(), quote!(default fn f<S: Simd>(simd: S) {})).is_err());
    }

    #[test]
    fn rejects_unsupported_function_attributes() {
        assert!(
            expand_err(quote!(
                #[track_caller]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("cannot preserve")
        );
        assert!(
            expand_err(quote!(
                #[naked]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("naked")
        );
        assert!(
            expand_err(quote!(
                #[unsafe(naked)]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("naked")
        );
        assert!(
            expand_err(quote!(
                #[instruction_set(arm::a32)]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("instruction_set")
        );
    }

    #[test]
    fn rejects_missing_or_unsupported_token_patterns() {
        assert!(
            expand_err(quote!(
                fn f() {}
            ))
            .contains("requires a SIMD token")
        );
        assert!(
            expand_err(quote!(
                fn f<S: Simd>(simd @ _: S) {}
            ))
            .contains("subpattern")
        );
        assert!(
            expand_err(quote!(
                fn f<S: Simd>((simd, _): (S, u32)) {}
            ))
            .contains("identifier or `_`")
        );
    }

    #[test]
    fn accepts_ref_token_binding() {
        expand_ok(quote! {
            fn f<S: Simd>(ref simd: S) -> S {
                let token: &S = simd;
                *token
            }
        });
    }

    #[test]
    fn accepts_ref_mut_token_binding() {
        expand_ok(quote! {
            fn f<S: Simd>(ref mut simd: S, replacement: S) -> S {
                let token: &mut S = simd;
                *token = replacement;
                *token
            }
        });
    }

    #[test]
    fn rejects_conditional_token_parameters() {
        assert!(
            expand_err(quote!(
                fn f<S: Simd>(#[cfg(any())] simd: S) {}
            ))
            .contains("cannot be conditional")
        );
        assert!(
            expand_err(quote!(
                fn f<S: Simd>(#[cfg_attr(any(), allow(unused))] simd: S) {}
            ))
            .contains("cannot be conditional")
        );
    }

    #[test]
    fn rejects_non_functions_and_bodyless_functions() {
        assert_eq!(
            expand_err(quote!(
                struct NotAFunction;
            )),
            "`#[simd]` can only be used on function and method definitions: expected `fn`"
        );
        assert_eq!(
            expand_err(quote!(
                fn bodyless<S: Simd>(simd: S);
            )),
            "`#[simd]` can only be used on function and method definitions: expected curly braces"
        );
    }
}
