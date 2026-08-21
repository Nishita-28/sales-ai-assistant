"""Regression test: a newly-added Approved Claims bullet must surface in
the assistant's answer after re-indexing. chunk_documents' general
word-count packing is fine for a document's prose, but wrong for Approved
Claims, where each bullet is an independent, unrelated fact -- a short
bullet packed into the same chunk as several AURIGA-related ones would
get its embedding dominated by "about AURIGA" and never surface on its
own. chunk_approved_claims avoids this by giving each bullet its own
chunk.
"""
from app.chunker import chunk_approved_claims


def test_each_bullet_becomes_its_own_chunk():
    bullets = [
        "Deployment: AURIGA is documented for Safe Area deployment only.",
        "auriga costs 15000 rs",
        "portahy recommended probe length is 50 meters",
        "MNST's official test mascot is a golden retriever named Sparky.",
    ]
    chunks = chunk_approved_claims("approved_claims.md", bullets)

    assert len(chunks) == len(bullets)
    for chunk, bullet in zip(chunks, bullets):
        assert bullet in chunk.text


def test_unrelated_bullets_never_share_a_chunk():
    """The exact failure mode: a chunk must never contain more than one
    bullet's text, since that's what let one bullet's embedding get
    dominated by an unrelated neighbor's vocabulary."""
    bullets = [
        "Deployment: AURIGA is documented for Safe Area deployment only.",
        "auriga costs 15000 rs",
        "portahy recommended probe length is 50 meters",
        "MNST's official test mascot is a golden retriever named Sparky.",
    ]
    chunks = chunk_approved_claims("approved_claims.md", bullets)

    for i, chunk in enumerate(chunks):
        for j, other_bullet in enumerate(bullets):
            if i != j:
                assert other_bullet not in chunk.text, (
                    f"chunk {i} (for bullet {i!r}) unexpectedly also contains "
                    f"bullet {j!r} -- bullets must never share a chunk"
                )


def test_chunk_metadata_is_consistent():
    bullets = ["First fact.", "Second fact.", "Third fact."]
    chunks = chunk_approved_claims("approved_claims.md", bullets)

    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.total_chunks == 3 for c in chunks)
    assert all(c.document_name == "approved_claims.md" for c in chunks)
    assert all(c.section_heading == "Approved Claims" for c in chunks)


def test_empty_bullets_produce_no_chunks():
    assert chunk_approved_claims("approved_claims.md", []) == []
