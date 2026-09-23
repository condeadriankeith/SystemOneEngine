---
name: fast-browser-check
description: Ultrafast, sub-second web and UI verification workflow using System One Engine and dynamic indexed DOM snapshots (based on jev-ultrafast architecture). Use when running browser tests, validating UI layouts, checking web endpoints, or performing automated browser interactions with minimal latency and zero screenshot overhead.
---

# Fast Browser & UI Check with System One Engine

This skill guides the agent to perform web and UI verification at ultra-fast speeds (<200ms per step) using **System One Engine** and the **`jev-ultrafast` dynamic indexed action space**.

## Core Philosophy: No Screenshots in the Decision Loop
Traditional VLM browser agents send multi-megabyte screenshots to vision models, consuming 1,500–4,000 tokens and 2–5 seconds per step.
This skill uses **structured, indexed DOM tables** and **sub-3ms System One Choice/Decide reflexes**.

---

## 1. The Dynamic Action Space

Capture the visible interactive page state atomically in **one browser evaluation**:
```text
[1] button    Sign In           · enabled
[2] textbox   Email Address     · empty
[3] textbox   Password          · empty
[4] button    Submit Form       · disabled
```

Every visible control is mapped to a numeric index. The agent never emits raw CSS selectors or coordinate clicks.

---

## 2. Decision Flow (Speculative Fan-out)

Use System One (`system_one_choice` or `/decide`):
1. **Operation**: `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_DOWN`, `WAIT`, `DONE`, `BLOCKED`.
2. **Target Head**: Integer index of the element corresponding to the chosen operation.
3. **Execution**:
   - For `CLICK` or `SELECT`: Dispatched immediately via CDP in <10ms.
   - For `TYPE_TEXT`: Dispatch to a small LLM with isolated field context to return strict JSON `{"text": "value"}`.

---

## 3. Pre-Execution Freshness & Occlusion Guard

Before firing any click:
1. Verify the element is still attached (`e.isConnected`).
2. Verify element is not occluded:
   ```javascript
   const r = e.getBoundingClientRect();
   const centerEl = document.elementFromPoint(r.x + r.width/2, r.y + r.height/2);
   if (!e.contains(centerEl)) throw new StalePage("Target occluded or animated");
   ```
3. If stale, re-observe before mutating state.

---

## 4. Verification & Completion
- Never treat a model's `DONE` selection as proof of success.
- Perform independent assertions (URL verification, DOM content check, network status).
