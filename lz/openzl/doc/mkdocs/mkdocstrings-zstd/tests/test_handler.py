from mkdocstrings_handlers.zstd.handler import ZstdHandler


def symbol_smoke_test(handler: ZstdHandler, name: str):
    item = handler.collect(name, {})
    html = handler.render(item, {})
    assert html


def test_handler_smoke(handler: ZstdHandler):
    symbols = [
        "MACRO1",
        "func1",
        "enum1",
        "s1",
        "u1",
        "g1",
        "x",
        "G1_MACRO",
        "g1_enum",
        "g1_struct",
        "g1_union",
        "g1_func",
        "func_in_para_returns",
        "var_list_items",
        "var_brief",
        "var_brief_and_detailed",
        "var_with_enumerated_list",
        "var_with_see",
    ]
    for symbol in symbols:
        symbol_smoke_test(handler, symbol)


def test_handler_empty_macro(handler: ZstdHandler):
    item = handler.collect("EMPTY_MACRO", {})
    html = handler.render(item, {})
    assert ">#define EMPTY_MACRO<" in html


def test_handler_simple_macro(handler: ZstdHandler):
    item = handler.collect("SIMPLE_MACRO", {})
    html = handler.render(item, {})
    assert ">#define SIMPLE_MACRO 1<" in html

    html = handler.render(item, {"show_define_initializer": False})
    assert ">#define SIMPLE_MACRO<" in html


def test_handler_func_macro(handler: ZstdHandler):
    item = handler.collect("FUNC_MACRO", {})
    html = handler.render(item, {})
    assert ">#define FUNC_MACRO(x) x<" in html

    html = handler.render(item, {"show_define_initializer": False})
    assert ">#define FUNC_MACRO(x)<" in html


def test_handler_func_macro_no_args(handler: ZstdHandler):
    item = handler.collect("FUNC_MACRO_NO_ARGS", {})
    html = handler.render(item, {})
    assert ">#define FUNC_MACRO_NO_ARGS() y<" in html


def test_handler_var_item_lists(handler: ZstdHandler):
    item = handler.collect("var_list_items", {})
    html = handler.render(item, {})
    assert "</ul>" in html
    assert "</li>" in html
    assert 'class="doc-section-title"' not in html


def test_handler_show_description(handler: ZstdHandler):
    item = handler.collect("var_brief", {})

    html = handler.render(item, {})
    assert "brief description" in html

    html = handler.render(item, {"show_description": False})
    assert "brief description" not in html


def test_handler_show_description_preconditions(handler: ZstdHandler):
    item = handler.collect("func1", {})

    html = handler.render(item, {})
    assert "Preconditions" in html
    assert "s != NULL" in html

    html = handler.render(item, {"show_description_preconditions": False})
    assert "Preconditions" not in html
    assert "s != NULL" not in html


def test_handler_show_description_postconditions(handler: ZstdHandler):
    item = handler.collect("func1", {})

    html = handler.render(item, {})
    assert "Postconditions" in html
    assert "s-&gt;x == x" in html

    html = handler.render(item, {"show_description_postconditions": False})
    assert "Postconditions" not in html
    assert "s-&gt;x == x" not in html


def test_handler_show_description_return(handler: ZstdHandler):
    item = handler.collect("func1", {})

    html = handler.render(item, {})
    assert "Returns" in html
    assert ">s-&gt;x<" in html

    html = handler.render(item, {"show_description_return": False})
    assert "Returns" not in html
    assert ">s-&gt;x<" not in html


def test_handler_func_with_param(handler: ZstdHandler):
    item = handler.collect("func_with_param", {})
    html = handler.render(item, {})
    print(html)
    pos = html.find('<div class="doc-md-description">')
    assert pos != -1
    html = html[pos:]
    assert html.count("<p>") == 1


def test_handler_func_with_code_block(handler: ZstdHandler):
    func = handler.collect("func_with_code_block", {})
    html = handler.render(func, {})
    print(html)
    assert '<pre><code class="language-cpp">' in html


def test_handler_enum(handler: ZstdHandler):
    item = handler.collect("enum1", {})
    html = handler.render(item, {})
    print(html)
    assert (
        '<span class="n">enum1_value1</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span>'
        in html
    )
    assert (
        '<div class="highlight"><pre><span></span><code><span class="n">enum1_value2</span>\n</code></pre></div>'
        in html
    )
    assert "The first enum value." in html
    assert "None" not in html


def test_typedef1(handler: ZstdHandler):
    item = handler.collect("typedef1", {})
    html = handler.render(item, {})

    print(html)
    assert (
        '<div class="highlight"><pre><span></span><code><span class="k">typedef</span><span class="w"> </span><span class="k">struct</span><span class="w"> </span><span class="nc">original</span><span class="w"> </span><span class="n">typedef1</span><span class="p">;</span>'
        in html
    )
    assert "This typedefs some original struct." in html


def test_var_with_enumerated_list(handler: ZstdHandler):
    item = handler.collect("var_with_enumerated_list", {})
    html = handler.render(item, {})
    assert "<ol>" in html
    assert "</ol>" in html
    assert "<ul>" not in html


def test_var_with_see(handler: ZstdHandler):
    item = handler.collect("var_with_see", {})
    html = handler.render(item, {})
    assert "See\n\n\n\nvar_with_enumerated_list which is a variable" in html


def test_func_that_throws(handler: ZstdHandler):
    item = handler.collect("ns5::func_that_throws", {})
    html = handler.render(item, {})
    print(html)
    assert "Parameters" in html
    assert "Exceptions" in html
    assert "std::runtime_error" in html
    assert "if x is negative" in html
