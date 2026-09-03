from safe._connect import analyze, ring_label

SPECIALS = frozenset(["[CLS]", "[SEP]", "[PAD]"])


def test_ring_label_parsing():
    assert ring_label("1") == 1
    assert ring_label("9") == 9
    assert ring_label("%12") == 12
    assert ring_label("%(100)") == 100
    assert ring_label("%(12345)") == 12345
    # bracket atoms / elements / bonds are not ring closures
    assert ring_label("[13C]") is None
    assert ring_label("[NH3+]") is None
    assert ring_label("[*:12]") is None
    assert ring_label("Cl") is None
    assert ring_label("C") is None
    assert ring_label("=") is None


def test_unstarted_prefix_is_not_complete():
    d = analyze(["C", "1"], [])
    assert not d.complete
    assert not d.started_completion
    assert d.open_label_count == 1
    assert d.current_attached  # the scaffold fragment is trivially "attached"


def test_decoration_closing_scaffold_label_is_complete():
    # scaffold "C1" (one open attachment), decoration ".C1" closes it
    d = analyze(["C", "1"], [".", "C", "1"])
    assert d.complete
    assert d.connected
    assert d.current_attached
    assert d.open_label_count == 0


def test_spurious_free_fragment_is_not_complete():
    # scaffold "C1" -> ".C1" attaches, then ".c2ccccc2" is a detached benzene
    gen = [".", "C", "1", ".", "c", "2", "c", "c", "c", "c", "2"]
    d = analyze(["C", "1"], gen)
    assert not d.complete  # the benzene is a separate component
    assert not d.connected
    assert not d.current_attached  # the fragment being written is detached


def test_attached_fragment_with_its_own_open_label_is_connected_but_incomplete():
    # decoration closes the scaffold label (attaches) but opens its own label
    d = analyze(["C", "1"], [".", "C", "1", "C", "2"])
    assert d.connected  # joined to the scaffold via label 1
    assert d.current_attached
    assert not d.complete  # label 2 is still open
    assert d.open_label_count == 1


def test_open_label_blocks_completion():
    # decoration opens its own label that is never closed
    d = analyze(["C", "1"], [".", "C", "1", "C", "2"])
    assert not d.complete
    assert d.open_label_count == 1


def test_unbalanced_parentheses_block_completion():
    d = analyze(["C"], ["C", "("])
    assert not d.complete
    # closing the branch makes it a valid single fragment
    d2 = analyze(["C"], ["C", "(", "C", ")"])
    assert d2.complete


def test_extended_ring_closure_labels_pair_across_fragments():
    # %(100) opened in the scaffold, closed by the decoration
    d = analyze(["C", "%(100)"], [".", "C", "%(100)"])
    assert d.complete
    assert d.open_label_count == 0


def test_multi_fragment_scaffold_prefix_is_one_component():
    # scaffold encoded as two fragments joined by a closed label 3, with an
    # open attachment label 1; decoration ".C1" closes the attachment.
    prefix = ["C", "3", "1", ".", "C", "3"]
    assert analyze(prefix, []).open_label_count == 1
    d = analyze(prefix, [".", "C", "1"])
    assert d.complete


def test_specials_are_ignored():
    d = analyze(["[CLS]", "C", "1"], [".", "C", "1", "[SEP]"], special_tokens=SPECIALS)
    assert d.complete
