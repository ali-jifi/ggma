"""One place that answers "what matcher and what prompts does this domain need?".

Its only caller today is safety/modal_safety_test/app.py::collect -- the safety side
runs on Modal, not locally. run_traces.py and smoke_test.py do NOT go through here;
they keep their own inline dispatch, and their safety paths are dead (both need a
`--classifier` that does not exist yet and a HarmBench CSV that is not in this tree).

So this is the seam, not a guarantee: a local driver that ever wants --domain safety
should route through here rather than re-deriving it, and should use load_safety --
run_traces.py's `load_harmbench` path assigns splits by hashing prompt_id, which
ignores the registry's extract/test provenance (Task 1.5). See load_safety.py.

Takes kwargs, not an argparse Namespace, so it is callable from a Modal function or a
test without faking an args object.
"""

HEURISTIC_SPEC = "heuristic"

_HEURISTIC_WARNING = """
!!! SCORER OF RECORD WARNING -- READ BEFORE USING THESE ROWS !!!

  Rollouts are being scored by HeuristicRefusalMatcher (regex over refusal
  prefixes), NOT the refusal classifier. Its own docstring: "Not the scorer of
  record; never substitute it for the classifier in production runs." The proposal
  keeps string matching "only as a sanity check", because it "overstates refusal and
  understates attack success".

  This is legitimate for ONE purpose: plumbing and pricing the pipeline while task
  0.3's safety-side conclusion matcher does not exist yet. Every row it scores is
  stamped scorer_version="refusal_heuristic-0.1.0".

  Piles built this way are DISCARDED, never promoted. store_rollout_texts is forced
  on so the rollouts can be re-scored offline once the real classifier lands.
"""


def build_matcher(domain, *, classifier=None, classifier_version="unversioned",
                  quiet=False):
    """The ConclusionMatcher for a domain.

    domain     : "math" | "safety"
    classifier : safety only. "module:callable" for the real classifier, or the
                 literal "heuristic" to opt IN to the regex baseline for plumbing.
                 Omitted -> raises, same as before: the scorer must be a choice.
    """
    if domain == "math":
        from matchers.gsm8k_matcher import GSM8KMatcher
        return GSM8KMatcher()
    if domain != "safety":
        raise ValueError(f"unknown domain {domain!r}")

    from matchers.refusal_matcher import (
        HeuristicRefusalMatcher,
        RefusalMatcher,
        load_classifier,
    )
    if not classifier:
        raise SystemExit(
            "safety domain needs --classifier.\n"
            "  --classifier module:callable   the refusal classifier (task 0.3, not built yet)\n"
            f"  --classifier {HEURISTIC_SPEC}          the regex baseline, for PLUMBING ONLY --\n"
            "                                 piles it scores must be discarded, not promoted."
        )
    if classifier == HEURISTIC_SPEC:
        if not quiet:
            print(_HEURISTIC_WARNING)
        return HeuristicRefusalMatcher()
    return RefusalMatcher(load_classifier(classifier), version=classifier_version)


def cross_matcher_for(domain, matcher):
    """The second matcher whose agreement with the scorer of record gets recorded as
    Trace.matcher_agreement_rate. refusal_matcher.py designates the heuristic for
    exactly this ("record base.agreement_rate(classifier_matcher, heuristic, texts)
    as matcher_agreement_rate on safety runs").

    None when there is nothing to cross-check against -- math, or when the heuristic
    IS the scorer (it cannot cross-check itself).
    """
    if domain != "safety":
        return None
    from matchers.refusal_matcher import HeuristicRefusalMatcher
    if isinstance(matcher, HeuristicRefusalMatcher):
        return None
    return HeuristicRefusalMatcher()


def load_prompts(domain, *, dataset=None, n=None, n_per_category=None, seed=0,
                 split=None, file_split=("extract", "test"), induction="natural",
                 preamble_id=None, local_gsm8k=None, hf_split="test",
                 allow_extraction_only=False, harmbench=None):
    """Prompt dicts for a domain.

    split : harness split to keep ("extraction"/"selection"/"heldout"), or None/"all".

    Safety goes through data.load_safety (the registry), NOT data.load_harmbench.
    load_harmbench wants a local CSV that does not exist in this tree, indexes its
    prompt_ids by row number, and calls assign_split -- which ignores the registry's
    extract/test provenance and so can score a prompt with a subspace built from it
    (Task 1.5). load_safety fixes all three. load_harmbench is left in place for the
    CSV path but is not what the drivers use.
    """
    split_filter = None if split in (None, "all") else split
    if domain == "math":
        from data.load_gsm8k import load_gsm8k
        return load_gsm8k(hf_split=hf_split, n=n, seed=seed, local_path=local_gsm8k,
                          split_filter=split_filter)
    if domain != "safety":
        raise ValueError(f"unknown domain {domain!r}")

    from data.load_safety import load_safety
    if not dataset:
        raise SystemExit(
            "safety domain needs --dataset <name> (a key in safety/datasets.py's "
            "REGISTRY, e.g. harmbench). Run `python3 safety/smoke_datasets.py` to see "
            "which sets are built."
        )
    return load_safety(dataset, n=n, n_per_category=n_per_category, seed=seed,
                       file_split=file_split, induction=induction,
                       preamble_id=preamble_id, split_filter=split_filter,
                       allow_extraction_only=allow_extraction_only)
