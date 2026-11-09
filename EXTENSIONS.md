# Possible extensions

These ideas are proposals, not features currently supplied by `sdf_ui`.

## Rule-guided document layout

Build an optional document-layout system that combines explicit rules, discrete
layout search, and the engine's differentiable SDF rendering and geometry
objectives. A first use case is turning a restricted Markdown document into an
editable, single-page report or brochure and a semantic PDF.

The goal is to let an author express requirements and preferences, inspect
candidate layouts and their rule explanations, lock decisions, and refine the
result. The system should make its choices reviewable rather than promising
that an optimized score automatically produces an attractive or accessible
document.

### Document and rule model

Represent the document as semantic nodes such as headings, paragraphs, lists,
images, captions, tables, and references. Preserve source locations, reading
order, links, language, alternative text, and author-specified constraints.
Measure text through a real shaping and typesetting implementation, including
font selection, glyph advances, line breaking, and hyphenation.

Rules should have stable identifiers, scopes, units, and explanations. Keep
three concepts separate:

* **Hard requirements** determine feasibility, for example no lost content,
  no content outside the printable region, and required captions and reading
  order. A finite penalty must not stand in for a guarantee.
* **Soft preferences** rank feasible layouts, for example balanced columns,
  consistent spacing, figure proximity, preferred margins, and whitespace.
  Their weights and normalization should be explicit.
* **Author locks** fix particular placement, size, or structure choices.

A diagnostic should name the affected elements and page, the measured value,
the requested threshold, and the rule responsible. When rules conflict, return
an actionable conflict report instead of silently publishing an invalid result.

### Hybrid layout pipeline

1. Parse a supported input into a semantic document and rule set.
2. Shape and measure text and other content at candidate widths.
3. Generate discrete candidates for column structure, line/page breaks, and
   figure placement using deterministic search or another appropriate solver.
4. Reject hard-rule violations and optimize continuous choices such as margins,
   spacing, column widths, and figure sizes within each viable structure.
5. Compare and explain candidates; provide an interactive preview and allow the
   author to adjust rules or lock choices.
6. Revalidate the chosen layout and export through a PDF/typesetting backend
   that preserves selectable text, links, structure, and accessibility tags.

Text reflows and page breaks are discontinuous. They require explicit discrete
handling and renewed measurement; a pixel gradient alone cannot optimize them
reliably. Use SDF images as previews and optional visual objectives, not as a
replacement for semantic text or a tagged PDF. A learned policy could later
propose candidate structures or starting values, but validation remains
independent of the policy.

### Integration with the current engine

The existing `Layout`, `Bounds`, and `Element` models, geometry losses,
`Objective.geometry` callbacks, render graph, and violation reporting provide a
starting point for fixed-structure candidate evaluation. They do not currently
provide a document object model, line shaping, pagination, a rule language,
discrete search, or a semantic PDF exporter. Text and glyph generation are not
differentiated by the native graph.

Use an established typesetter such as Typst or TeX for text handling and PDF
export if an adapter can obtain the measurements and placement control the
prototype needs. Verify the chosen integration's actual APIs and semantics;
do not assume it exposes internal line or page layout data. An initial adapter
may treat the typesetter as a black box and score rendered candidates, with
the associated limitations and cost made explicit.

### Suggested stages and validation

1. **Single page:** support a restricted Markdown subset with headings,
   paragraphs, one image, captions, and a footer. Implement rule diagnostics,
   an SDF preview, and export through a real typesetter.
2. **Candidate search:** compare a small set of column and image-placement
   structures, then optimize their continuous parameters. Support locked
   placements and explain infeasible rule sets.
3. **Multipage documents:** add page breaking, figure placement, tables,
   widow/orphan checks, incremental reflow, and a reproducible candidate search.
4. **Optional learning:** train a proposal policy only after collecting suitable
   examples and defining held-out evaluation. Keep deterministic rules and
   exact validation as the final gate.

Test against documents with long and short text, varied page sizes, multiple
languages and fonts, missing assets, infeasible constraints, and discontinuous
reflows. Verify determinism, semantic reading order, text extraction, links,
tagged PDF behavior, and visual rendering. Compare layout quality, latency, and
rule violations against the chosen typesetter's ordinary output; do not assume
the optimizer is better without evidence.
