# Goal Description

The goal is to make the entire frontend application responsive for mobile devices, and introduce a "Dark Mode" using a CSS inversion trick as requested ("invert the white clrs into black"). We will also add a dark mode toggle button in the user's Profile page.

## User Review Required

> [!WARNING]
> **CSS Inversion technique for Dark Mode**
> The `filter: invert(1) hue-rotate(180deg)` approach is a quick way to achieve a dark mode. It will turn `#fafafa` (light grey) into `#050505` (almost black), but it can sometimes cause minor artifacts on shadows, borders, or colorful gradients. We will also invert images back so they look normal. Please confirm if you are okay with this quick CSS trick or if you'd prefer a full Tailwind dark mode (which would take significantly longer).

## Open Questions

None currently, but please review the approach above!

## Proposed Changes

### Global Styles & State

#### [MODIFY] [index.css](file:///c:/Users/kunal/nirvexa-frontend/src/index.css)
- Add `.dark-theme` CSS rules that apply `filter: invert(1) hue-rotate(180deg)` to the HTML body.
- Add counter-inversion for `img`, `video`, and specific UI elements (like colored gradients or avatars) so their original colors are preserved.

#### [MODIFY] [App.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/App.jsx) (or new Context)
- We will add a global state (persisted to `localStorage`) to track whether Dark Mode is enabled.
- The state will toggle the `.dark-theme` class on the `<html>` or `<body>` element.

### Pages & Responsiveness

#### [MODIFY] [Profile.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/Profile.jsx)
- Add a "Dark Mode" toggle button/switch in the Profile settings.
- Improve grid layouts to ensure they collapse into single columns on small screens (`flex-col` instead of `flex-row` on mobile).

#### [MODIFY] [Layout.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/components/layout/Layout.jsx) & Other Pages
- Review and apply mobile-first Tailwind utilities (`sm:`, `md:`, `lg:`) to containers.
- Fix hardcoded widths like `w-[500px]` that overflow on mobile screens.
- Ensure padding and font sizes are appropriately scaled down on small devices.

## Verification Plan

### Automated/Manual Tests
- I will run the frontend dev server (`npm run dev`) in the background if possible, or request that you preview it.
- Toggle the dark mode button in the Profile page to verify the CSS inversion works and doesn't break images.
- Shrink the browser window / use mobile view to verify the layout of key pages (Profile, Landing, SkillMatch) doesn't overflow horizontally.
