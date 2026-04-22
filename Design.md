# Design System Document: High-End Editorial 2048

## 1. Overview & Creative North Star
**Creative North Star: "The Tactile Gallery"**

This design system moves away from the "plastic" feel of traditional mobile games and toward a sophisticated, editorial experience. We treat the 2048 grid not as a digital toy, but as a curated collection of objects resting on a premium surface.

To break the "template" look, we employ **intentional asymmetry** in the layout—placing the score and utility controls in an off-balance, editorial arrangement—and use high-contrast typography scales (the bold, architectural Manrope paired with the functional Inter) to create a sense of authority and intentionality.

---

## 2. Colors & Surface Logic
The palette is rooted in soft neutrals (`surface`) to provide a "breathable" canvas, allowing the tile progression to feel like a blooming spectrum of heat and energy.

### The "No-Line" Rule
**Explicit Instruction:** Designers are prohibited from using 1px solid borders to section the UI. Boundaries must be defined solely through background color shifts. Use `surface-container-low` for the main page background and `surface-container-highest` for the game board itself to create a recessed, "carved" look without a single stroke line.

### Surface Hierarchy & Nesting
Treat the UI as a series of stacked premium materials:
- **Base Level:** `surface` (The gallery wall).
- **Secondary Level:** `surface-container-low` (The display plinth).
- **Interactive Level:** `surface-container-lowest` (Floating cards for "New Game" or "Settings").

### The "Glass & Gradient" Rule
To add visual "soul," use subtle linear gradients for the tiles. A tile should not be a flat `#8c4a00`; it should transition from `primary` to `primary_container` at a 135-degree angle. For floating overlays (like a "Game Over" screen), use a **Glassmorphism** approach:
- **Fill:** `surface` at 70% opacity.
- **Effect:** 20px Backdrop Blur.
- **Outcome:** The vibrant colors of the game board bleed through the menu, creating a high-end, integrated feel.

---

## 3. Typography
We use a dual-font strategy to balance character with legibility.

* **Display & Headlines (Manrope):** This is our "Architectural" voice. Use `display-lg` for the "2048" logo and `headline-md` for game states. The wide apertures of Manrope feel modern and premium.
* **Interface & Body (Inter):** This is our "Functional" voice. Use `title-md` for tile numbers and `label-sm` for secondary data (like "Best Score" labels). Inter’s tall x-height ensures readability even when tiles are small on mobile.

**Hierarchy Tip:** Never use the same weight for a label and its value. Use `label-sm` in `on_surface_variant` for the "SCORE" text, and `title-lg` in `on_surface` for the actual number.

---

## 4. Elevation & Depth
In this system, depth is a product of **Tonal Layering**, not structural shadows.

* **The Layering Principle:** Place a `surface-container-lowest` card on top of a `surface-container` background to create a soft, natural lift. This mimics the way a sheet of heavy cardstock sits on a table.
* **Ambient Shadows:** If an element must float (e.g., a modal), use a "Sunken Shadow":
* `box-shadow: 0 24px 48px -12px rgba(46, 47, 45, 0.08);`
* The shadow color is derived from `on_surface` at a very low opacity to mimic natural ambient light.
* **The "Ghost Border" Fallback:** If a tile needs more definition against the board, use a "Ghost Border": `outline-variant` at 15% opacity. Never use 100% opaque borders.

---

## 5. Components

### The Game Tile (The Hero)
- **Shape:** Use `md` (0.75rem) roundedness.
- **Color:** Map the tile value to the intensity of the palette. Low values (2, 4) use `tertiary_container`. Mid-values (16, 32, 64) use `primary_fixed`. High values (128+) use `secondary`.
- **Motion:** Tiles should scale from 0.9 to 1.0 on spawn with a "spring" easing to feel tactile.

### Buttons (Actionable Logic)
- **Primary:** `primary` background with `on_primary` text. Use `full` (9999px) roundedness for a friendly, "pebble" feel.
- **Tertiary (Ghost):** No background. Use `on_surface` text with `label-md` styling. Perfect for "Undo" or "Rules."

### Score Modules
- **Layout:** A vertical stack. A `label-sm` (all caps, tracked out +5%) sits above a `title-lg` value.
- **Container:** `surface_container_high` with `sm` roundedness.

### Forbid the Divider
**Rule:** Forbid the use of horizontal rules (`
`). Use the Spacing Scale (e.g., `8` or `10`) to create "Active Negative Space" between the header and the game grid.


---

## 6. Do's and Don'ts

### Do
* **DO** use `surface_bright` for the "New Game" button to make it the clear primary call-to-action.
* **DO** use the `16` (4rem) spacing token for the top margin to give the header "room to breathe"—editorial design is about the luxury of space.
* **DO** ensure tiles with dark backgrounds (like `primary`) use `on_primary` for the text color to maintain accessibility.

### Don't
* **DON'T** use pure black `#000000` for text. Always use `on_surface` (`#2e2f2d`) to maintain the "soft minimalist" tone.
* **DON'T** use `none` roundedness. Even the grid should have at least `lg` (1rem) roundedness to feel approachable.
* **DON'T** use standard "Drop Shadows" from a library. If it looks like a default Bootstrap shadow, it is wrong for this system. Use the Ambient Shadow formula.