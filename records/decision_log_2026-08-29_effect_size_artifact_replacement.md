# Decision log: effect-size artifact replacement

Date: 2026-08-29  
Decision: ISSUE VERSIONED V2 REPLACEMENT; DO NOT FABRICATE V1

The frozen protocol references
`baseline_platform/artifacts/effect_size_decision_freeze_v1.json` with SHA-256
`e312669f0901b386d439de6374c22624f06bbb51a7f52b4878cd7ed0b05a818c`,
but that file is not present in the current workspace. A workspace-wide filename,
hash, and terminology search did not locate another copy.

The missing file is not recreated under the V1 name and its historical hash is
not claimed. Instead, `effect_size_decision_freeze_v2.json` records the exact
thresholds already written explicitly in the hash-frozen protocol v0.3. This is
an audit-chain repair, not a change to H1, H2, H3, the statistical plan, data
roles, or any test lock.

No training or prediction generation was performed to make this decision. The
V2 replacement does not authorize train/validation execution and does not
authorize test or external-test access.

