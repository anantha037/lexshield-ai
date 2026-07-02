"""
Bug 3 - Demonstration: strip_reasoning_preamble() + _is_malformed_response() examples
=======================================================================================
Run from repo root:  python rag/scripts/demo_reasoning_strip.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.multi_llm import strip_reasoning_preamble, _is_malformed_response

SEP = "-" * 72

# --- Example 1: Clean answer - should pass through completely untouched -------
CLEAN_ANSWER = """\
**Relevant Law:** Section 302 of the Indian Penal Code (IPC) [1]

**What it says:** Section 302 prescribes the punishment for the offence of
murder. A person who commits murder shall be punished with death or
imprisonment for life, and shall also be liable to a fine. [1]

**Answer:** Yes, a conviction under Section 302 IPC carries either the death
penalty or life imprisonment -- the court decides based on the nature of the
offence and mitigating circumstances. [1][2]

**Punishment/Remedy:** Death or life imprisonment, plus fine. [1]"""

# --- Example 2: Reasoning leak - <think> block at start (DeepSeek R1 style) --
LEAKED_THINK = """\
<think>
The user is asking about murder under IPC. Let me think through this:
- Section 302 IPC = murder, punishable by death or life imprisonment
- BNS equivalent is Section 103 (2023 reform, effective July 1 2024)
- I have [SOURCE 1] which covers Section 302 and [SOURCE 2] for Section 103
I should structure the answer with old law and new law both cited.
</think>

**Relevant Law:** Section 302 of the Indian Penal Code (IPC) [1] and
Section 103 of the Bharatiya Nyaya Sanhita (BNS) [2]

**What it says:** Under the old law (pre-July 2024): Section 302 IPC makes
murder punishable by death or life imprisonment plus fine. [1]
Under the new law (post-July 2024): Section 103 BNS carries the same
punishment. [2]

**Answer:** A conviction for murder carries death or life imprisonment under
both IPC (pre-July 2024) and BNS (post-July 2024). [1][2]"""

# --- Example 3: <reasoning> tag variant (some Qwen3 thinking builds) ----------
LEAKED_REASONING = """\
<reasoning>
Query involves tenant eviction rights. Key statutes to consider:
  - Transfer of Property Act 1882, Section 106 (notice period)
  - Rent Control Act (state-specific -- user hasn't specified state)
  - Article 300A Constitution (right to property)
No specific state mentioned so I'll note state-law dependency.
Retrieved [SOURCE 1] covers notice requirements. [SOURCE 2] is Article 300A.
</reasoning>

**Your Rights:** As a tenant, your landlord cannot evict you without
following due legal process. [SOURCE 1] Key protections include:
- A mandatory notice period before eviction proceedings begin [SOURCE 1]
- The right not to be forcibly dispossessed without a court order [SOURCE 2]

**Relevant Law:** Section 106 of the Transfer of Property Act 1882 requires
the landlord to give at least 15 days' written notice before termination. [SOURCE 1]"""

# --- Example 4: Prose opener -- NOT stripped (deliberate conservative choice) -
# "Let me think through this clause" could be a legitimate opener on NDA queries.
LEGIT_PROSE_OPENER = """\
Let me think through the key provisions of this non-disclosure agreement.

**Clause 3 -- Confidentiality Obligation:** This clause is standard and
enforceable under the Indian Contract Act 1872. [1]

**Clause 7 -- Penalty:** A penalty of Rs 10 lakh for each breach is
enforceable only if it is a genuine pre-estimate of loss, not a penalty
clause designed to terrify. [1][2] Courts may reduce it under Section 74 ICA."""

# --- Example 5: Section 26 MVA-style untagged prose reasoning (the actual -----
# production incident). No XML tags -- strip_reasoning_preamble() correctly
# leaves this untouched (conservatism by design). _is_malformed_response()
# is the second layer that catches it.
UNTAGGED_MVA_PROSE = """\
Let me analyze the Motor Vehicles Act, Section 26 and what the retrieved sources tell us.

The query is about requirements for a driving licence under the Motor Vehicles Act.
Let me think through what Source 1 says about this.

Looking at [SOURCE 1], Section 26 deals with the grant of driving licences to applicants
who have held a learner's licence for at least 30 days. The conditions are laid out clearly.
The applicant must pass a driving test conducted by the licensing authority.

Based on my analysis of the retrieved sources, here is the final answer:

Under Section 26 of the Motor Vehicles Act 1988, a permanent driving licence may be
granted to a person who has held a learner's licence for a minimum period of 30 days
and has passed the driving test administered by the licensing authority. [SOURCE 1]"""

# ===============================================================================
# Run all five strip_reasoning_preamble() cases
# ===============================================================================
strip_cases = [
    ("1 - Clean legal answer (no CoT tag)",
     CLEAN_ANSWER, False),
    ("2 - <think> preamble leak (DeepSeek R1 style)",
     LEAKED_THINK, True),
    ("3 - <reasoning> preamble leak (Qwen3 style)",
     LEAKED_REASONING, True),
    ("4 - Prose opener 'Let me think...' (NOT stripped -- too ambiguous)",
     LEGIT_PROSE_OPENER, False),
    ("5 - Untagged prose reasoning leak (Section 26 MVA incident)",
     UNTAGGED_MVA_PROSE, False),   # strip_reasoning_preamble: NO strip expected
]

print("=" * 72)
print("LAYER 1: strip_reasoning_preamble() -- tagged CoT blocks only")
print("=" * 72)

all_pass = True
for label, raw, expect_strip in strip_cases:
    result = strip_reasoning_preamble(raw)
    did_strip = result != raw.strip()
    ok = "PASS" if did_strip == expect_strip else "FAIL <-- UNEXPECTED"
    if did_strip != expect_strip:
        all_pass = False
    print()
    print(SEP)
    print(f"Example {label}")
    print(f"Expected strip: {expect_strip} | Actually stripped: {did_strip}  [{ok}]")
    print(SEP)
    if did_strip:
        print("[INPUT - first 180 chars including reasoning block]")
        print(repr(raw[:180]))
        print()
        print("[OUTPUT - after strip_reasoning_preamble()]")
        print(result[:400])
    else:
        if label.startswith("5"):
            # Show the full untagged prose so the gap is visible
            print("[OUTPUT - UNCHANGED (gap: no XML tag, so layer 1 cannot strip)]")
            print(result[:500])
        else:
            print("[OUTPUT - unchanged (first 300 chars shown)]")
            print(result[:300])

print()
print("Layer 1 overall:", "ALL PASS" if all_pass else "SOME FAILURES")

# ===============================================================================
# Layer 2: _is_malformed_response() -- untagged prose detection
# ===============================================================================
print()
print("=" * 72)
print("LAYER 2: _is_malformed_response() -- untagged prose + missing header")
print("=" * 72)
print("(AND logic: meta-commentary signal in first 300 chars AND no synthesis")
print(" header in first 150 chars.  Both must hold to flag as malformed.)")
print()

malformed_cases = [
    # (label, text_to_check, expect_malformed)
    ("1 - Clean structured answer",
     strip_reasoning_preamble(CLEAN_ANSWER), False),
    ("2 - After strip: <think> answer (tagged CoT removed)",
     strip_reasoning_preamble(LEAKED_THINK), False),
    ("3 - After strip: <reasoning> answer (tagged CoT removed)",
     strip_reasoning_preamble(LEAKED_REASONING), False),
    ("4 - Prose opener with header after first line",
     strip_reasoning_preamble(LEGIT_PROSE_OPENER), False),
    ("5 - Untagged MVA prose (NOT stripped by layer 1) -- caught here",
     strip_reasoning_preamble(UNTAGGED_MVA_PROSE), True),  # EXPECTED: malformed
]

layer2_pass = True
for label, text, expect_malformed in malformed_cases:
    result = _is_malformed_response(text)
    ok = "PASS" if result == expect_malformed else "FAIL <-- UNEXPECTED"
    if result != expect_malformed:
        layer2_pass = False
    print(SEP)
    print(f"Example {label}")
    print(f"Expected malformed: {expect_malformed} | Detected: {result}  [{ok}]")
    if label.startswith("5"):
        print("  -> generate() will: _record_failure() + continue to next provider")
        # Explain the two conditions for transparency
        window = text[:300].lower()
        from rag.multi_llm import _META_COMMENTARY_SIGNALS, _SYNTHESIS_HEADERS
        hit_signals = [s for s in _META_COMMENTARY_SIGNALS if s in window]
        start = text[:150].lower().lstrip()
        hit_headers = [h for h in _SYNTHESIS_HEADERS if start.startswith(h)]
        print(f"  Condition 1 (meta-commentary signals found): {hit_signals}")
        print(f"  Condition 2 (synthesis header at start): {hit_headers or 'none'}")

print()

# ===============================================================================
# Citation marker survival check (on tagged-CoT cases)
# ===============================================================================
print(SEP)
print("Citation-marker survival check (tagged-CoT examples 2 and 3)")
print(SEP)
result2 = strip_reasoning_preamble(LEAKED_THINK)
for marker in ["[1]", "[2]", "[1][2]"]:
    present = marker in result2
    status = "OK" if present else "LOST -- BUG"
    print(f"  {marker!r:12s} in output: {present}  [{status}]")
    if not present:
        all_pass = False

result3 = strip_reasoning_preamble(LEAKED_REASONING)
for marker in ["[SOURCE 1]", "[SOURCE 2]"]:
    present = marker in result3
    status = "OK" if present else "LOST -- BUG"
    print(f"  {marker!r:12s} in output: {present}  [{status}]")
    if not present:
        all_pass = False

print()
print(SEP)
final_ok = all_pass and layer2_pass
print("Overall:", "ALL PASS" if final_ok else "SOME FAILURES -- review above")
print(SEP)
