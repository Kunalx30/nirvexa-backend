# Plan: Professional Job Page Redesign, Logo Integration, and Deployment Prep

This plan outlines the design enhancements to the job page, replacing the text/square logo placeholder with the newly added logo asset, pointing the production environment to the backend API URL, and pushing the final code to GitHub.

## User Review Required

> [!IMPORTANT]
> **API URL Verification**: We will point the production environment (`.env.production`) to the custom backend URL `https://api.nyrvexa.in/api`. Please ensure your Render backend custom domain is active.
> **Logo Asset**: The new logo file `file_000000008d9471fabb37e610b2a9bb34.png` in `public` has been copied to a standard, clean filename `logo.png`. We will use this image across the application.

## Proposed Changes

---

### 1. Logo Asset Replacement (Frontend)

We will replace the text-based square logo `<div className="*lsq">N</div>` with the new logo image.

#### [MODIFY] [Navbar.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/components/layout/Navbar.jsx)
- Replace `<div className="nb-lsq">N</div>` with an image tag pointing to `/logo.png`.

#### [MODIFY] [Landing.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/Landing.jsx)
- Replace logo icons in header and footer with image tags pointing to `/logo.png`.

#### [MODIFY] [Login.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/Login.jsx)
- Replace the logo placeholder in the auth card with `/logo.png`.

#### [MODIFY] [Register.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/Register.jsx)
- Replace the logo placeholder in the register card with `/logo.png`.

#### [MODIFY] [NotFound.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/NotFound.jsx)
- Replace the logo placeholder in the 404 card with `/logo.png`.

#### [MODIFY] [Pricing.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/Pricing.jsx)
- Replace the logo placeholder in the pricing page header with `/logo.png`.

#### [MODIFY] [UserAdmin.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/UserAdmin.jsx)
- Replace the logo placeholder in the admin auth card and sidebar header with `/logo.png`.

---

### 2. Job Page Polish & Redesign (Frontend)

We will polish [Jobs.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/Jobs.jsx) to make it look highly professional, matching premium pages like the Career Path page.

#### [MODIFY] [Jobs.jsx](file:///c:/Users/kunal/nirvexa-frontend/src/pages/Jobs.jsx)
- Add a subtle dotted/grid background to the page container.
- Implement soft animations (fade-in/fade-slide) when the job cards load.
- Enhance card aesthetics using clean borders, premium shadows, and micro-hover zoom effects.
- Style the search container and filters to utilize polished glassmorphism/inset highlights.
- Align typography and layout sizes perfectly.

---

### 3. Production Environment Target Configuration (Frontend)

#### [MODIFY] [.env.production](file:///c:/Users/kunal/nirvexa-frontend/.env.production)
- Uncomment `VITE_API_URL=https://api.nyrvexa.in/api`.
- Comment out/remove the local development fallback `http://127.0.0.1:5000/api`.

---

### 4. Push Code to GitHub

We will commit all modified and created files across the backend and frontend repositories and push them to their respective GitHub remotes.

## Verification Plan

### Manual Verification
- Run `npm run build` locally in `nirvexa-frontend` to verify that there are no syntax/build errors after modifying files.
- Inspect the file modifications locally using a git diff command.
- Push the commits and verify they are successfully pushed to GitHub.
