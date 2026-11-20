// Where the related-articles popup opens for its "···" trigger. Pure, so the rule is tested without a browser.
type Box = { left: number; top: number; right: number; bottom: number };

const GAP = 8, EDGE_X = 16, EDGE_Y = 12;

/**
 * Under the trigger when it fits, else beside it, else above it, always inside the viewport. Only when no side has room
 * (a small phone) does the popup cover its trigger; `coversAnchor` reports that, because a second press on the same
 * spot then lands on the popup instead of the trigger.
 */
export function placePopup(anchor: Box, width: number, height: number, viewport: { width: number; height: number }) {
  const maxLeft = viewport.width - width - EDGE_X, maxTop = viewport.height - height - EDGE_Y;
  const aligned = Math.max(EDGE_X, Math.min(anchor.left, maxLeft)), level = Math.max(EDGE_Y, Math.min(anchor.top, maxTop));
  const below = Math.max(EDGE_Y, anchor.bottom + GAP), above = anchor.top - GAP - height;
  const after = anchor.right + GAP, before = anchor.left - GAP - width;
  const at = (left: number, top: number) => ({ left, top, coversAnchor: left < anchor.right && left + width > anchor.left && top < anchor.bottom && top + height > anchor.top });

  if (below <= maxTop) return at(aligned, below);
  if (after <= maxLeft) return at(after, level);
  if (before >= EDGE_X) return at(before, level);
  if (above >= EDGE_Y) return at(aligned, above);
  return at(aligned, Math.max(EDGE_Y, maxTop));
}
