# Ethics & responsible use

This framework evaluates how visual manipulations affect age estimation models. Because
that work necessarily involves images of children, please read this before running it.

## What this framework will and will not do

**It will:**
- Provide loaders, manipulation pipelines, model wrappers, and evaluation metrics.
- Help you quantify how often a model misclassifies a minor as an adult under various
  appearance edits.

**It will not:**
- Bundle any face images, including UTKFace.
- Generate or distribute manipulated images of identifiable real children.
- Provide tooling to bypass any specific deployed age verification system.

## Your responsibilities

If you run this framework on a dataset containing images of minors:

1. **Verify the dataset license.** UTKFace is non-commercial research only. Other
   datasets may require institutional access.
2. **Check institutional ethics policies.** Most academic settings require IRB or
   ethics-board review for research involving child images, even when the dataset
   is publicly available.
3. **Limit access to manipulated images.** The output of the manipulation pipeline
   includes modified images of minors. Treat these as sensitive: do not commit them
   to public repos, do not distribute them outside your research team, and delete
   them when the project ends.
4. **Be careful about what you publish.** Publish aggregate metrics and example
   manipulations on adult/synthetic faces only. Do not include manipulated images of
   identifiable children in papers, slides, or blog posts.
5. **Consider downstream harm.** Results from this framework can inform two very
   different audiences: defenders trying to harden age verification, and adversaries
   trying to bypass it. Disclose findings responsibly and consider coordinating with
   affected vendors before publication.

## What the framework checks

The framework itself enforces a few mechanical safeguards:

- The headline metric is the *minor → adult* misclassification rate, framed as a
  failure to protect children. The framing is intentional: this is a tool for
  understanding shortcomings of age verification, not a tool for evading it.
- The classical manipulation pipeline is deterministic and uses landmark-aligned
  geometric overlays. No identity-preserving generative editing is applied unless
  you explicitly enable the GenAI pipeline (`manipulations.genai.enabled: true`).
- The framework writes results to local disk only; no telemetry, no remote calls
  except those needed to download model weights.

If you have feedback on these safeguards, please open an issue.
