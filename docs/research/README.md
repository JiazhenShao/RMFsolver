# RESEARCH MEMORY INDEX — READ BEFORE WORKING ON THE NUCLEAR–QUARK FRONT

This directory contains the current research conclusions and implementation handoffs for the hydrodynamical nuclear-to-quark conversion project.

## Current authoritative notes

1. [`THERMAL_INTERFACE_TEMPERATURE_CLOSURE__CURRENT_CONCLUSION.md`](THERMAL_INTERFACE_TEMPERATURE_CLOSURE__CURRENT_CONCLUSION.md)
   - Current conclusion on using thermal transport to close the sharp interface.
   - Corrects the distinction between thermal diffusivity and flavor diffusion.
   - Records the Wolfram-verified formulas and literature sources.

2. [`LTE_BOUND_TO_SIGNED_TEMPERATURE_JUMP_SCAN__2026-07-23_HANDOFF.md`](LTE_BOUND_TO_SIGNED_TEMPERATURE_JUMP_SCAN__2026-07-23_HANDOFF.md)
   - Handoff for the planned two-block adaptive scan.
   - Defines the requested observable `delta_T_interface = T_N - T_Qstar_LTE`.
   - Identifies the exact older cluster runner that should be used as the structural template.

## Superseded note

`docs/superpowers/specs/2026-07-22-thermal-interface-jump-bound-design.md` is superseded.
Its claim that thermal and flavor relaxation share the same low-temperature `T^(-5/3)` transport time is not consistent with the Heiselberg–Pethick thermal-relaxation result.
Do not implement that specification without first reconciling it with the current conclusion recorded here.

