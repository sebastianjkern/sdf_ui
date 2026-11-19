# sdf-ui implementation skills

These repository skills capture recurring implementation and review work for
sdf-ui. Use the feature skill for API placement and scope, the learning skill
whenever parameters or optimization are involved, and the shader lifecycle skill
for GPU passes, caches, or texture ownership.

| Skill | Use it for |
| --- | --- |
| [Engine feature](engine-feature/SKILL.md) | New primitives, effects, public API, and reusable widgets |
| [Learning support](learning-support/SKILL.md) | Making low-level and high-level components trainable |
| [Shader lifecycle](shader-lifecycle/SKILL.md) | Shader implementation, resource lifetime, cache correctness, and repeated rendering |

For a feature that spans layers, apply the engine feature skill first, then the
learning and shader lifecycle skills wherever their triggers apply.
