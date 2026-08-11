# 1. Eraser model with explicit planes replaces layout-bounds hull

Date: 2026-08-11

## Status

Accepted

## Context

The annotation tool masked AprilTags by projecting the axis-aligned bounding box of all marker footprints, taking the convex hull, and pasting a background plate. That erases more than the stickers and cannot mask non-convex or partial object regions accurately.

## Decision

- Store eraser geometry in `config/Model/<object>/eraser_model.json`, separate from `marker_model.json`.
- Each **Eraser Plane** is a quad with named corners (`top_left`, `top_right`, `bottom_right`, `bottom_left`).
- Corner values are offsets from the **Reference Marker Center** in the **Layout Frame** (meters).
- The eraser model declares `"origin": "reference_marker_center"`.
- Each frame: project all planes with the **Fused Object Pose**, clip partially visible polygons to the image, union into one **Eraser Mask**, paste the **Background Plate**.
- Remove the layout-bounds convex hull path entirely.

## Consequences

- Authors can define multiple planes (e.g. front rubber, back, edges) without over-erasing.
- Eraser authoring matches marker footprint style but uses relative offsets from the reference marker center.
- `marker_model.json` stays detection-only per object profile; eraser geometry can change without affecting pose.
- Partially visible planes still erase their visible portion (clip, do not skip).
