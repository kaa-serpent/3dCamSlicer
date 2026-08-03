# Makera Z1 community measurement status

This sheet records the supplied observations without assigning them stronger CNC
meaning than the measurement method supports. The bundled profile remains unverified,
preview-only and ineligible for export.

## Recorded observations

| Observation | Recorded value | How RotaryCAM uses it |
|---|---:|---|
| Controller firmware shown by UI | `1.0.4Beta5` | Preserved as a user-reported UI observation; it does not certify G93 or simultaneous XYZA. |
| Firmware image on controller storage | `FIRMWARE.CUR`: `1.0.4Beta2`, build `Sep 16 2025 10:53:01` | Preserved separately because it disagrees with the UI report. Neither value silently replaces the other. |
| Active controller configuration | `config.txt`, SHA-256 `7B2DA495BAEF31D18AF484D2FAB363C03FA4562377159EF89F8DC81855A9F0C1` | Authoritative for the controller parameters recorded below. |
| Axis maximum rates | X/Y `2000`, Z `1000` mm/min; A `1800` deg/min | Populates the typed X/Y/Z/A dynamics and the conservative linear/rotary speed caps. |
| Axis accelerations | X/Y/Z `150` mm/s²; A `360` deg/s² | Populates the typed X/Y/Z/A dynamics. |
| Default seek rate | `2000` mm/min | Recorded in the config snapshot. It does not bypass rapid-path collision validation. |
| Configured XY work area | `200 × 200` mm | Informational controller configuration. It is not converted into G54 Y travel. |
| Enabled soft-endstop minima | MCS X `-210`, Y `-212`, Z `-105` mm | Stored explicitly in machine coordinates (MCS), not as G54 `MachinePose` limits. |
| Anchor 1 | MCS X `-191.55`, Y `-193.639` mm | Provenance for interpreting displayed coordinates; it is not a G54 origin contract. |
| Controller rotation offsets | X `-7.5`, Y `69`, Z `23` | Stored verbatim and informationally. In particular, `rotation_offset_z = 23` is not `rotary_pivot_z`. |
| Home screen display | X `190.550`, Y `192.639`, Z `69.343` mm | Informational metadata only; the coordinate frame and relationship to G54 travel limits are not established. |
| Coordinate display | 3 decimal places (`0.000`) | Display formatting only; it is not positioning accuracy, repeatability or resolution evidence. |
| Reported rotary mount position | X `60`, Y `69` mm | Informational metadata only; its coordinate frame and the rotary pivot's required G54 Y/Z coordinates remain unknown. |
| Reported rotary behavior | `A CW = Y+` | Preserved verbatim. Without a fixed viewing direction and a marked point's start/end position, it does not resolve RotaryCAM's `direction` sign. |
| Spindle-nose/quick-change envelope | diameter `16` mm, length `23` mm | Editable prefill for each tool assembly. It is not a separate physical holder and is never accepted automatically as measured. |
| Tool stickout | Variable after every tool change/calibration | Must be measured per tool from the tool tip to the spindle-nose face before collision-safe export validation. |

`MachineObservationMetadata` keeps the MCS values and firmware provenance separate from
G54 geometry. The rates and accelerations declared by the active `config.txt` are also
copied into typed dynamics, but the work area, soft limits, Anchor 1 and rotation offsets
remain observations only. Planner and postprocessor safety checks must not reinterpret
those observations as G54 motion contracts.

## Still blocking export

- Proven X/Y/Z travel limits expressed in the same G54 coordinate system as generated
  `MachinePose` values.
- Rotary pivot Y/Z in G54, an unambiguous positive-A direction test and the setup transform.
- Complete conservative primitives for chuck, jaws, tailstock, platter, spindle and supports.
- Controller evidence and a controlled no-stock dry-run for simultaneous XYZA and G93
  inverse-time behavior. The UI reports `1.0.4Beta5`, while `FIRMWARE.CUR` identifies
  itself as `1.0.4Beta2`; the active runtime version still requires a read-only live query.
- Per-tool stickout confirmation and review of the 16 mm diameter × 23 mm conservative
  nose envelope.

## File identities

- `config.txt`: SHA-256
  `7B2DA495BAEF31D18AF484D2FAB363C03FA4562377159EF89F8DC81855A9F0C1`.
- `FIRMWARE.CUR`: SHA-256
  `3608C6FBDB6C568A7098464C9DDAC8129469853EA44B8C75A18F766074C99515`.

The files were inspected read-only. They are not bundled into RotaryCAM and must never be
written back to the controller by profile loading or editing.
