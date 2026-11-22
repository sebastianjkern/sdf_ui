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
