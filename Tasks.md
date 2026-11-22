Tasks:

- **Small:** Make built-in plugin loading atomic. If a plugin registration
  raises, `PluginRegistry.ensure_loaded()` has already set `_loaded`, so
  subsequent lookups see a partially populated registry and report
  unrelated plugins as missing. The root cause is in
  `src/sdf_ui/gpu/nodes/registry.py`.

- **Small:** Resolve the existing Slop duplicate code failure for the hard
  Boolean GPU plugin registrations. Intersection, subtraction, and union repeat
  the same registration structure in their respective
  `src/sdf_ui/gpu/nodes/primitives/{intersection,subtract,union}/plugin.py`
  modules. Use a shared registration helper while preserving each shader
  interface and differentiation contract.

- **Medium:** Bring the existing full tree Complexipy scan under the hook's
  limit. Branch-heavy logic in `_font_bidi.py`, `fonts.py`,
  `core/texture.py`, `gpu/spatial.py`, `gpu/tiled.py`, and the polyline and
  SVG primitive plugins exceeds the limit. Split responsibilities within each
  function and preserve its existing behavior.

- **Small:** Apply the Ruff formatter across `src/`, `build_tools/`, and
  `tests/` in a formatting-only change. The full-tree formatter check reports
  174 files outside Ruff's current formatting conventions. CI checks only a
  subset of those directories.

- **Small:** Clarify pen-up speed control in `src/sdf_ui/export.py`.
  `travel_speed_mm_min` is emitted as an `F` word on `G0` moves, but GRBL and
  LinuxCNC use controller-configured rapid rates for `G0`. An explicit
  feed-controlled travel mode would make the configured speed effective.

- **Medium:** Clear existing Slop findings in `examples/alpine_engraving.py`.
  `_hatch` combines clipping and mark generation in nested loops, while
  `_make_drawing` combines composition data and drawing logic. The hatch and
  lighting code also embeds unnamed numeric tuning values. Preserve the
  default seed's render when refactoring.

- **Small:** Correct the pre-existing Ruff import-order failure in
  `examples/__main__.py`. The `hand_hatch_example` import precedes
  `half_plane_example`, contrary to Ruff's alphabetical ordering.
