# Build the Nexar welcome / home screen

You are building a single marketing-style welcome page for **Nexar**, an internal conversational analytics application built on Databricks Genie. The page's job is to make a senior, non-technical audience understand the product in about forty seconds of looking at it, without reading much.

Build it to match this specification exactly. Where I give pixel values, colours or timings, use them literally rather than approximating.

---

## 1. Stack

- Match the existing app's stack. If it is React, build it as a single page component with co-located CSS (CSS modules or styled-components — follow whatever the repo already uses). If there is no existing convention, produce one self-contained HTML file with inline `<style>` and `<script>`.
- No CSS framework, no component library, no animation library. Plain CSS transitions/keyframes and vanilla DOM measurement only.
- No `localStorage`, `sessionStorage` or any browser storage.
- All SVG icons inline. Do not add an icon package.

## 2. Fonts

Load from Google Fonts:

- **Plus Jakarta Sans** (500/700/800) — all headings, card titles, anything with class `b`/`h1`–`h4`. Letter-spacing `-0.025em`.
- **Inter** (400/500/600/700) — all body text. Base size 15px, line-height 1.65.
- **JetBrains Mono** (400/500) — small uppercase labels, code, IDs, counts.

## 3. Design tokens

Declare these as CSS custom properties on `:root` and use them everywhere. Do not introduce colours outside this list.

```css
--ink:#141728;      --ink2:#3f4459;     --muted:#6b7185;
--line:#e4e7f0;     --wash:#f7f8fc;     --wash2:#eef0f7;
--chatbg:#f1f2f9;   --nav:#22243c;
--indigo:#4f46e5;   --indigo-l:#eef1fe; --indigo-d:#3730a3;
--violet:#8b5cf6;   --green:#10b981;    --amber:#f59e0b;
--grey:#a3a8b8;     --red:#dc4f4f;
--sh:0 1px 2px rgba(20,23,40,.04),0 10px 30px rgba(20,23,40,.06);
```

**Colour semantics — enforce these strictly:**
- Indigo is the primary action and the default agent colour.
- Violet is used only for the ambiguity/semantic stage.
- **Green means verified, confident, or in-scope, and nothing else.** Never use it as a decorative accent.
- Red appears exactly once, on the out-of-scope callout.
- Amber is the brand mark and the "certified prompts" marker.

Two content widths: `.wrap` is `max-width:1180px; padding:0 30px`. `.wrap.wide` overrides to `max-width:1440px`.

---

## 4. Page structure

Five blocks, in order:

1. Top navigation bar
2. Hero (headline, subhead, two CTAs)
3. **Showcase** — a recreation of the real app UI with seven annotation callouts pinned to it
4. Agents section — six clickable cards with an expanding detail panel
5. Final CTA

### 4.1 Top nav

Full-width, `background:var(--nav)`, height 66px, white text, inside `.wrap`:
- Left: hamburger (three 1.5px bars, `#c9ccda`, 20px wide, 4px gap), then a 22px circle with a 2.5px `--amber` border containing an 8px amber dot, then the wordmark "Nexar" at 19px/700 Plus Jakarta Sans.
- Centre (`margin:0 auto`): pill, `background:#2e3152`, `border:1px solid #3d4168`, radius 999px, padding `6px 15px`, 12.5px text `#d5d8e6`, reading "Powered by Databricks Genie", preceded by a 6px green dot that pulses via an expanding box-shadow keyframe on a 2.2s loop.
- Right: "BETA" in an 11px bordered chip.

### 4.2 Hero

Centred, `padding:60px 0 0`, background `linear-gradient(180deg,#f0f0ff 0%,#f8f8fd 46%,#fff 100%)`, plus a `::before` radial glow (`920×540`, `rgba(99,102,241,.26)`, `blur(60px)`) positioned `top:-190px; left:50%; translateX(-50%)`.

- **H1**: `clamp(30px,4.3vw,50px)`, weight 800, line-height 1.05, `max-width:17ch`. Text: **"Your data, answered in a sentence."**
  Animate it word by word: split into spans, each word wrapped in an `overflow:hidden` span containing an `<i>` that starts at `translateY(110%)` and rises to 0 over 0.95s with `cubic-bezier(.16,1,.3,1)`, staggered 55ms per word.
- **Subhead**: `--muted`, 17px, `max-width:50ch`, fades in at 0.5s. Text: *"A governed semantic layer on Databricks. Ask in your own words, get a number you can defend."*
- **CTAs** (fade in at 0.68s): primary "Browse spaces" — indigo fill, radius 10px, padding `13px 25px`, `box-shadow:0 10px 28px rgba(79,70,229,.3)`. Secondary "See the agents" — white with `--line` border, anchors to `#agents`.

---

## 5. The showcase (the important part)

A `position:relative` container inside `.wrap.wide`, `margin-top:52px`, `padding-bottom:92px`. It holds three layers:

- `<svg id="links">` — absolutely positioned, `inset:0`, `z-index:5`, `pointer-events:none`, starts `opacity:0`.
- `.app` — the UI recreation, `z-index:2`.
- `.noteshost` — seven absolutely positioned callout cards, `z-index:6`.

### 5.1 The app frame

`max-width:900px`, centred, `background:#fff`, `border:1px solid #d9dced`, `border-radius:14px`, `box-shadow:0 34px 80px rgba(28,24,88,.2)`, `overflow:hidden`. Fades in at 0.5s.

**App title bar** (44px, `background:var(--nav)`): miniature of the top nav — small hamburger, amber ring mark, "Nexar" at 14px, centred Genie pill at 10px, BETA chip at 9px.

**App body**: `display:grid; grid-template-columns:52px 1fr 244px; min-height:560px`.

#### Column 1 — icon rail

White, `border-right:1px solid var(--line)`, `padding:16px 0`, centred column, 17px gap. Six 28px icon buttons at `color:#8a90a6`, 16px stroke-1.8 SVGs:

1. **Plus** — new chat. Give this one `color:var(--indigo)`.
2. **Four-square grid** — dashboards.
3. *(1px × 24px divider, `--line`)*
4. **Clock** — history.
5. **Cube** — model selection. Needs `id="a-models"`.
6. **Dollar sign** — tokenomics.
7. **Circular arrow** — refresh.

#### Column 2 — chat

`background:var(--chatbg)`, flex column, `border-right:1px solid var(--line)`.

**Header** (`padding:14px 17px`, 12px gap): a white "← Back" chip with indigo 12px/600 text; the room title `id="a-title"` at 15.5px/700 Plus Jakarta Sans reading **"Renewals Caseflow Management"**; then `margin-left:auto` a "+ New chat" indigo button, radius 8px, 11.5px/600.

**Body** (`flex:1; padding:2px 17px 8px`, 14px gap, `overflow:hidden`) containing three messages:

**(a) User message** — `id="a-q"`. Indigo fill, white text, `border-radius:12px`, `padding:11px 14px`, `align-self:flex-end`, `max-width:76%`. Inside: a meta row with "YOU" at 9.5px/700, `letter-spacing:.09em`, `rgba(255,255,255,.72)` on the left and "04:44 PM" at 10px same colour on the right (`justify-content:space-between`); below it the message at 14px. The text is typed in character by character (see timeline) with a 1.5px blinking caret.

**(b) Ambiguity response** — `id="clar"`. A label row: "NEXAR · AMBIGUITY AGENT" at 9.5px/700, `letter-spacing:.09em`, `color:var(--indigo)`, with "04:44 PM" muted on the right. Below, a card `id="a-clar"`: white, `border:1px solid #e6dcfb`, `border-left:3px solid var(--violet)`, radius 10px. Title **"Fiscal or calendar quarter?"** (13px/700), body *"Two certified calendars match "last quarter" here."* (12px muted), then two option chips — **"Fiscal Q2 FY26"** selected (violet fill, white text) and **"Calendar Q2 2026"** unselected (white, `--line` border).

**(c) Answer** — `id="ans"`. Label row "NEXAR ANSWER" + "04:45 PM". Card: white, `--line` border, radius 10px, `padding:13px 14px`.
- Title: **"Renewal rate, EMEA — down 6.2 points"** (14px/700).
- Description: *"Q2 FY26. Concentrated in mid-market accounts renewing after the May price change."* (12px muted).
- Confidence pill `id="a-conf"`: inline-flex, `background:#ecfdf5`, `border:1px solid #c5eddc`, `color:#047857`, radius 999px, `padding:3px 9px`, 10.5px/600, a 6px green dot then **"94% confidence · renewal_rate v4"**.
- Action buttons at `margin-left:auto`: two 26px bordered icon buttons (chevron-down, circular-refresh) then a small indigo "Full screen" button with an expand icon.
- Chart `id="a-chart"`: `margin-top:10px`, `border-top:1px solid var(--line)`, `padding-top:10px`. An inline SVG line chart, `viewBox="0 0 340 66"`, two faint gridlines at y=17 and y=43, a gradient area fill under the line (`--indigo` at 0.16 fading to 0), and the line itself declining left to right, 2.2px indigo, round caps and joins. **The line draws itself**: set `stroke-dasharray:420; stroke-dashoffset:420` and transition the offset to 0 over 1.5s with a 0.2s delay when the parent gets `.in`.

**Footer** (`padding:8px 17px 16px`): a pill input bar — white, `--line` border, `border-radius:999px`, height 45px, `padding:0 6px 0 19px`, `box-shadow:0 2px 8px rgba(20,23,40,.05)`. Placeholder text at 12.5px `#9aa0b4` reading *"Message Genie in Renewals Caseflow Management…"*, then `margin-left:auto` a 33px indigo circle with a white paper-plane. Give the bar `id="a-input"`.

#### Column 3 — context panel

White, `padding:16px 14px`. Header "CONTEXT PANEL" at 12.5px/700 Plus Jakarta Sans with a bottom border.

Four sections, each `padding-bottom:16px; margin-bottom:16px; border-bottom:1px solid var(--line)` (last one has none). Each section header is a flex row: a 10px/700 uppercase label with `letter-spacing:.08em`, then `margin-left:auto` a count badge (`background:var(--indigo-l)`, `color:var(--indigo-d)`, radius 999px, `padding:1px 8px`, 10px/600) and a small `▾` caret.

1. **`id="a-prompts"` — CERTIFIED PROMPTS**, badge `3`, plus a small search icon before the badge and a 10px muted subtitle *"Click to edit, double click to send"*. Then three rows, each `background:var(--wash)`, `--line` border, radius 7px, `padding:6px 9px`, 11px: "Renewal rate by segment", "Top SLA breaches this week", "Accounts pending information".
2. **`id="a-tools"` — TOOLS**, badge `1`. A pill chip: `background:var(--indigo-l)`, `border:1px solid #d5daf9`, radius 999px, JetBrains Mono 10.5px, `color:var(--indigo-d)`, a small activity-line icon then `@charts`. Below it, 10px muted: *"Visualize query results as a chart"*.
3. **`id="a-dashsec"` — DASHBOARDS**, badge `1`. One item with `border-left:2.5px solid var(--indigo)`, `background:var(--wash)`, `border-radius:0 7px 7px 0`: title "Renewals Caseflow - Sigma" (11px), subtitle "Embedded Sigma dashboard for Rene…" (10px muted, truncated with ellipsis).
4. **`id="a-links"` — SOURCE LINKS**, badge `2`. Two rows, each a chain-link icon, a truncating two-line label ("renewal_global_dashbo…" / "Primary BYOD dataset for R…" and "colt_launchpad" / same), then an "Open" button outlined in `#c9cef1` with indigo text.

### 5.2 The seven callouts

Absolutely positioned cards inside `.noteshost`. Each: `width:216px`, white, `--line` border, `border-radius:13px`, `padding:13px 15px`, `box-shadow:0 16px 38px rgba(28,24,88,.14)`, `opacity:0` until revealed.

Card contents: a `lab` row (JetBrains Mono 9px, muted, with a 6px coloured dot), a `b` title (12.5px), and a `span` body (11px, muted, line-height 1.55).

Each carries `data-anchor` (a CSS selector for the UI element it describes) and `data-side` (`left` or `right`).

| # | Side | `top` | `data-anchor` | Label / dot | Title | Body |
|---|---|---|---|---|---|---|
| 1 | left | 56px | `#a-models` | "models & tokenomics" / indigo | Choose the model per agent | Per-room model selection, with token spend tracked as you go. |
| 2 | left | 262px | `#a-clar` | "ambiguity agent" / violet | It asks before it guesses | Two calendars match. Asked once, then remembered for the room. |
| 3 | left | 468px | `#a-input` | "ask anything in scope" / red | It stops rather than guessing | Named individuals and unmodelled data get a boundary, not a guess. |
| 4 | right | 34px | `#a-prompts` | "certified prompts" / amber | Questions already signed off | Vetted per room by the data team, ready to run. |
| 5 | right | 186px | `#a-tools` | "tools" / indigo | Agents you can call by name | @charts turns a result into the visual that fits it. |
| 6 | right | 338px | `#a-dashsec` | "dashboards" / green | Open one inside the chat | Sigma dashboards embed in the thread. No tab switching. |
| 7 | right | 490px | `#a-links` | "source links" / green | Walk back to the tables | Every answer names the datasets behind it. |

Left cards use `left:0`, right cards use `right:0`. Card 3 additionally gets a `warn` modifier: `border-color:#f2dada; background:#fefafa;` and its title in `#a83c3c`.

Three on the left and four on the right is deliberate — the context panel has four naturally spaced anchors, the chat column only has three. Do not add a fourth left callout; it produces converging connector lines.

### 5.3 Connector lines — implement carefully

Do **not** hardcode line coordinates. Write a `drawLinks()` function that measures real geometry:

```
for each callout note:
  tgt = document.querySelector(note.dataset.anchor)
  nb  = note.getBoundingClientRect()
  tb  = tgt.getBoundingClientRect()
  hb  = showcaseContainer.getBoundingClientRect()
  right = note.dataset.side === 'right'

  x1 = (right ? nb.left : nb.right) - hb.left      // inner edge of the card
  y1 = nb.top - hb.top + nb.height / 2
  x2 = (right ? tb.right + 4 : tb.left - 4) - hb.left   // outer edge of the target
  y2 = tb.top - hb.top + Math.min(tb.height / 2, 32)    // cap so tall targets anchor near their top
  mx = (x1 + x2) / 2

  emit: <path d="M x1 y1 C mx y1, mx y2, x2 y2"
              fill="none" stroke="#c3c8e6" stroke-width="1.4" stroke-dasharray="4 4"/>
        <circle cx=x1 cy=y1 r="2.6" fill="#c3c8e6"/>                                   // card end
        <circle cx=x2 cy=y2 r="3.6" fill="#fff" stroke="#4f46e5" stroke-width="1.8"/>  // target end
```

Set the SVG's `width`, `height` and `viewBox` from the container rect first. Then add `.on` to fade it in.

Redraw on `resize`, debounced by 150ms. Bail out early (clear the SVG, remove `.on`) if `.noteshost` computes to `display:none`.

### 5.4 Animation timeline

Everything is time-based from load. Exact values:

| t (ms) | Event |
|---|---|
| 1000 | User bubble fades/slides in (`opacity` + `translateY(8px)`, 0.45s) |
| 1300 | Typing begins — one character every **32ms** |
| +500 after typing ends | Ambiguity response fades in (0.5s) |
| +1500 after typing ends | Answer card fades in; the chart line starts drawing |
| +2200 after typing ends | Callouts begin popping in, **150ms apart**, each 0.62s `cubic-bezier(.19,1,.22,1)` from `translateY(14px) scale(.95)` |
| +400 after the last callout | `drawLinks()` runs and the SVG fades in |

The ordering matters: annotations must not appear before the thing they annotate exists.

---

## 6. Agents section (`id="agents"`)

Centred heading **"Six agents built that answer"** (`clamp(25px,3.1vw,35px)`, weight 800) and subhead *"Each runs on every question, on a model you choose. Open any of them."*

**Grid**: `repeat(3,1fr)`, 14px gap. Each card is a `<button>`: white, `--line` border, `border-radius:16px`, `padding:20px`, `box-shadow:var(--sh)`.
- Top row: a 28px circle with a 2px coloured border containing the agent number in JetBrains Mono, the name at 15.5px/700, and a `→` at `margin-left:auto`.
- Below: a one-line description at 13.5px muted.
- Border colour per card: agents 1 and 4 indigo, 2 and 3 violet, 5 and 6 green.
- Hover: `translateY(-3px)`, `box-shadow:0 16px 38px rgba(24,20,80,.1)`, border `#d6d9ee`, arrow shifts 4px right, plus a cursor-following radial glow — a `::before` with `background:radial-gradient(280px circle at var(--mx) var(--my), rgba(79,70,229,.08), transparent 62%)` fading in, with `--mx`/`--my` set from a `pointermove` handler.
- Selected: `background:var(--ink)`, white text, muted text `#a9aec4`, arrow rotated 90°.

Card copy:

1. **Intent mapping** — Classifies the question before anything is queried.
2. **Ambiguity** — Asks one clarifying question instead of guessing.
3. **Semantic resolution** — Maps your words onto certified entities and metrics.
4. **Query generation** — Runs it in your workspace, under your permissions.
5. **Charting** — The @charts tool picks the visual that fits.
6. **Confidence & trace** — Scores the mapping and names the source tables.

**Detail panel**: hidden by default, appears below the grid when a card is clicked, `margin-top:16px`, white, `--line` border, `border-radius:18px`, `box-shadow:0 18px 44px rgba(24,20,80,.09)`, animating in from `translateY(-8px)` over 0.45s. Inner layout `grid-template-columns:1fr 1fr; gap:44px; padding:34px 36px`.

Left half: a pill "Agent N · Name" with the agent's dot colour; an `h3`; a paragraph at 16px muted; and a "Close" button. Right half: a `.frag` panel (`background:var(--wash)`, `--line` border, radius 14px, padding 22px) holding a bespoke visual per agent.

Clicking the already-open card closes the panel.

### Detail content

**1 — Intent mapping**
Heading: *First — what kind of question is this?*
Body: *Nothing is queried until the question has been classified. The shape decides everything downstream: a trend needs a time series, a ranking needs a comparison, a lookup needs neither.*
Visual: four classification rows, each a label with a progress bar and a right-aligned percentage. "Trend with a causal drill-down" 92% (bold, bar filled with `linear-gradient(90deg,var(--indigo),var(--violet))`), "Ranked comparison" 4%, "Single value lookup" 3%, "Forecast" 1% (grey bars). Bars animate from width 0 to their value over 1s when the panel opens.

**2 — Ambiguity**
Heading: *Then it refuses to guess*
Body: *"Last quarter" is genuinely ambiguous here — two certified calendars match it. Rather than picking one silently and being quietly wrong, Nexar asks. The answer is stored against the room, so the team is asked once, not every time.*
Visual: reuse the clarifier card component, plus a note below with a 3px indigo left border: *"Answered once by the room owner. Every later question here resolves "last quarter" the same way."*

**3 — Semantic resolution**
Heading: *Three phrasings, one certified number*
Body: *Nobody has to learn a schema. However someone asks it, the question lands on the definition the data team signed off — not on whichever table looked closest.*
Visual: three-column grid — a stack of three white chips reading `"churn"`, `"lapsed"`, `"non-renewed"`; a `→`; and a bordered indigo chip reading `renewal_rate` in mono with the caption "certified · v4 · Finance Data".

**4 — Query generation**
Heading: *The query runs where the data already lives*
Body: *Executed inside your Databricks workspace under the permissions you already hold. Nexar keeps no copy of your data, and it cannot widen what someone can already see.*
Visual: a dark code block (`background:#1b1e33`, radius 11px, JetBrains Mono 12px) with syntax colouring — comment `#6b7290`, keywords `#a5b4fc`, strings `#6ee7b7`:
```sql
-- built from the certified metric, not the raw tables
SELECT period, renewal_rate
FROM metrics.renewal_rate_v4
WHERE region = 'EMEA'
  AND fiscal_quarter = 'Q2 FY26'
```
Then three green-outlined chips: "Unity Catalog permissions", "Runs in your workspace", "Logged for audit".

**5 — Charting**
Heading: *The chart is chosen, not defaulted*
Body: *It runs as the @charts tool in the context panel — automatically on most answers, or on demand when you call it. The visual follows the shape of the question, so splitting by segment changes the chart with it.*
Visual: three small chart-type tiles with inline SVG glyphs — a line ("Time series", selected: indigo border and `--indigo-l` background, caption "chosen"), bars ("Bar", "not used"), a donut ("Composition", "not used"). Below, a bordered note: *"The question asks why a number moved. Movement over time is a line — a bar chart would hide the trajectory that makes the answer useful."*

**6 — Confidence & trace**
Heading: *And it hands you the reasoning*
Body: *The mapping is scored on the answer itself, and the datasets behind it are listed under source links — so anyone can walk back from the number to the tables it came from.*
Visual: a six-row numbered trace list (numbers in 25px bordered circles, indigo for 1 and 4, violet for 2 and 3, green for 5 and 6):
1. Intent read as a trend with a causal drill-down
2. Ambiguity raised — resolved to `Q2 FY26`
3. "renewals" resolved to `renewal_rate` v4
4. Query run on `renewal_global_dashboard`
5. Chart type chosen: time series
6. Mapping scored `0.94` and stored

Below a divider, a 70px conic-gradient ring (`conic-gradient(var(--green) 338deg, #e4e7f1 0)`) with a white inner circle showing "94%", beside the text: *"Confidence reflects how cleanly the question mapped onto a certified metric — not how sure the model feels about the number."*

---

## 7. Final CTA

Centred. `h2` **"Fourteen governed rooms are live"** at `clamp(26px,3.4vw,38px)` weight 800, `max-width:19ch`, then the primary indigo "Browse all spaces" button.

---

## 8. Responsive

**≤1420px** — Hide the SVG connectors and `.noteshost` entirely. Render the same seven callouts instead as a static 4-column grid below the frame (`max-width:900px`, `margin:28px auto 0`, 14px gap), fully visible with no animation. Build this by cloning the callouts' inner HTML into a `.notesgrid` container at runtime so the copy is never duplicated in source.

**≤900px** — App body collapses to a single column; hide the icon rail and context panel. Agent grid becomes 2 columns. Detail panel becomes one column, `padding:26px 22px`. Callout grid becomes 2 columns. `.wrap` padding drops to 18px.

**≤600px** — Agent grid, callout grid and chart-type tiles all become single column. Hide the "Full screen" button's text label, keeping the icon.

## 9. Accessibility

- Respect `prefers-reduced-motion: reduce`: disable every animation and transition, and render the end state directly — question text fully typed, all messages and callouts visible, chart line at `stroke-dashoffset:0`, connectors drawn.
- Agent cards are real `<button>` elements, keyboard operable, with `:focus-visible { outline:2px solid var(--indigo); outline-offset:3px; border-radius:9px }`.
- The showcase SVG gets `aria-hidden="true"` — it is decorative; the callout text carries the meaning.
- Callout cards are readable in DOM order after the frame.

## 10. Acceptance criteria

- [ ] Every `data-anchor` selector resolves to a real element in the DOM.
- [ ] Connector endpoints land on the target elements at 1440px, 1600px and 1920px viewport widths, and after a window resize.
- [ ] No connector line crosses the app frame; left callouts only target chat-column elements, right callouts only target context-panel elements.
- [ ] Callouts appear only after the answer card has rendered.
- [ ] Below 1420px no lines render and the callouts appear as a grid, with no duplicated markup in the HTML source.
- [ ] Green appears only on confidence, in-scope and verified elements.
- [ ] With reduced motion enabled the page is fully readable and complete with zero animation.
- [ ] No console errors; no browser storage APIs used.

## 11. Data to replace before this ships

Everything below is placeholder. Wire it to real values, or confirm the real ones with me:

- Room name, timestamps, and the message content
- "94% confidence", "renewal_rate v4", and the "down 6.2 points" figure
- The three certified prompt names
- The two source link names and descriptions
- The dashboard name in the DASHBOARDS section
- "Fourteen governed rooms" in the final CTA
- The generated SQL in agent 4 — make it match what the product actually emits
- Whether `@charts` is the only tool, and whether all six agents are actually live. If any are still in build, mark them rather than presenting them as shipped.
