# Plan: Rename Website to Orbital-AI and Implement the Site

## Context

The project contains a detailed design specification at `src/imports/pasted_text/satquery-ai-website-design.md` describing a premium dark-themed satellite intelligence website originally branded as "SatQuery AI". The `src/App.tsx` component is currently an empty shell. The user wants the brand name changed to "Orbital-AI" and the website built.

## Changes

### 1. Update design spec (`src/imports/pasted_text/satquery-ai-website-design.md`)
- Line 1: `"SatQuery AI"` → `"Orbital-AI"`
- Line 139: `SatQuery AI logo` → `Orbital-AI logo`
- Line 142: `"Try SatQuery"` → `"Try Orbital-AI"`

### 2. Implement the website in `src/App.tsx`

Build the full page described in the spec with Orbital-AI branding. All sections below go into `src/App.tsx` (and any supporting component files if needed to stay within complexity limits).

**Color tokens** (from spec):
- `#07111F` — deep space navy (primary bg)
- `#0D1B2A` — midnight blue (secondary sections)
- `#20D9FF` — electric cyan (primary accent)
- `#FF9F43` — solar orange (secondary accent)
- `#35E0B8` — mint teal (status/data)
- `#F4F7FA` — soft white (primary text)
- `#9AA9B8` — cool gray (secondary text)
- `#101F30` — blue-black (cards/panels)

**Sections to implement:**

1. **Navbar** — `Orbital-AI` logo · Explore · Analyze · Features · About · "Upload Image" (outlined) · "Try Orbital-AI" (cyan CTA)

2. **Hero** — Asymmetric layout: left has label "AI-POWERED EARTH INTELLIGENCE", headline "Ask Anything About Earth.", supporting copy, cyan primary CTA, outlined secondary CTA; right has a satellite imagery visualization with coordinate markers, detection boxes, data labels, terrain classification, floating query interface.

3. **Product Interface Section** — Large dashboard preview: satellite image viewer with zoom/layer controls, AI chat panel, query input, detection overlays, and statistics tiles (land classification, building count, water coverage, vegetation %).

4. **Feature Sections** — Varied compositions (not a repetitive grid): large visual + small supporting features, split-screen layouts, data visualization, before/after comparison. Show real query examples like "How much of this area is covered by water?"

5. **Data Visualization** — Clean charts with cyan lines, orange comparison indicators, mint positive trends on dark-blue panels.

6. **Footer** — Orbital-AI branding, links, tagline.

**Typography:**
- Headings: geometric sans-serif (Inter or similar via Google Fonts)
- Body: highly readable sans-serif
- Small uppercase technical labels and coordinate microtext throughout

**Styling approach:**
- All styling via Tailwind CSS v4 utility classes
- Subtle grain/noise texture via CSS (SVG filter or pseudo-element)
- Geographic/grid pattern via CSS background-image
- Controlled glow effects via `box-shadow` utilities or inline style where needed
- Scroll-based reveals via Intersection Observer + CSS transitions
- Google Fonts (`Inter` + `Space Grotesk`) added via `@import` at the top of `src/index.css`

## Files to Modify

- `src/imports/pasted_text/satquery-ai-website-design.md` — 3 text replacements
- `src/App.tsx` — full implementation of the site
- `src/index.css` — add Google Fonts `@import` and base body color

## Verification

1. Dev server is already running; preview panel should immediately reflect changes.
2. Check that "Orbital-AI" appears in the navbar logo and the CTA button ("Try Orbital-AI").
3. Verify no "SatQuery" text remains anywhere visible.
4. Confirm all sections render without console errors.
