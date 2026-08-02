# Makera Z1 community measurement status

This sheet records the supplied observations without assigning them stronger CNC
meaning than the measurement method supports. The bundled profile remains unverified,
preview-only and ineligible for export.

## Recorded observations

| Observation | Recorded value | How RotaryCAM uses it |
|---|---:|---|
| Controller firmware | `1.0.4Beta5` | Informational metadata only; it does not certify G93 or simultaneous XYZA. |
| Home screen display | X `190.550`, Y `192.639`, Z `69.343` mm | Informational metadata only; the coordinate frame and relationship to G54 travel limits are not established. |
| Coordinate display | 3 decimal places (`0.000`) | Display formatting only; it is not positioning accuracy, repeatability or resolution evidence. |
| Reported rotary mount position | X `60`, Y `69` mm | Informational metadata only; its coordinate frame and the rotary pivot's required G54 Y/Z coordinates remain unknown. |
| Reported rotary behavior | `A CW = Y+` | Preserved verbatim. Without a fixed viewing direction and a marked point's start/end position, it does not resolve RotaryCAM's `direction` sign. |
| Spindle-nose/quick-change envelope | diameter `16` mm, length `23` mm | Editable prefill for each tool assembly. It is not a separate physical holder and is never accepted automatically as measured. |
| Tool stickout | Variable after every tool change/calibration | Must be measured per tool from the tool tip to the spindle-nose face before collision-safe export validation. |

`MachineObservationMetadata` is intentionally separate from axis limits,
`XYZAConfiguration`, dynamics, capabilities and measured assembly geometry. Planner and
postprocessor safety checks do not consume these observations as motion contracts.

## Still blocking export

- Proven X/Y/Z travel limits expressed in the same G54 coordinate system as generated
  `MachinePose` values.
- Rotary pivot Y/Z in G54, an unambiguous positive-A direction test and the setup transform.
- X/Y/Z/A maximum velocity and acceleration measurements.
- Complete conservative primitives for chuck, jaws, tailstock, platter, spindle and supports.
- Controller evidence and a controlled no-stock dry-run for simultaneous XYZA and G93
  inverse-time behavior on firmware `1.0.4Beta5`.
- Per-tool stickout confirmation and review of the Ø16 × 23 mm conservative nose envelope.
