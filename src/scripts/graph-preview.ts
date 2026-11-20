// Document-level delegation: the same handlers serve the article rail and the dialog whose graph is injected later.
const isSelected = (node: SVGAElement) => node.getAttribute('aria-current') === 'true';

function graphNode(target: EventTarget | null) {
  return target instanceof Element ? target.closest<SVGAElement>('.graph-node') : null;
}

function select(node: SVGAElement) {
  const graph = node.closest<HTMLElement>('.graph-wrap');
  const card = graph?.querySelector<HTMLAnchorElement>('.graph-card');
  if (!graph || !card) return;

  for (const other of graph.querySelectorAll('.graph-node[aria-current]')) other.removeAttribute('aria-current');
  node.setAttribute('aria-current', 'true');

  const { title = '', type = '', publishedAt = '', reason = '', via = '' } = node.dataset;
  const part = <T extends HTMLElement = HTMLElement>(name: string) => card.querySelector<T>('.graph-card-' + name)!;
  card.href = node.getAttribute('href') || '';
  part('type').textContent = type;
  part<HTMLTimeElement>('date').dateTime = part('date').textContent = publishedAt;
  part('title').textContent = title;
  part('title').lang = node.getAttribute('lang') || '';
  part('via').textContent = via;
  part('via').hidden = !via;
  part('reason').textContent = reason;
}

export function measureRelatedDialogHeight(dialog: HTMLDialogElement) {
  const nodes = [...dialog.querySelectorAll<SVGAElement>('.graph-node')];
  const selected = nodes.find(isSelected);
  let height = dialog.getBoundingClientRect().height;
  if (!selected) return height;

  // Reserve room for the tallest card when positioning the popup, so a later selection never pushes it off-screen.
  for (const node of nodes) {
    select(node);
    height = Math.max(height, dialog.getBoundingClientRect().height);
  }
  select(selected);
  return height;
}

// A tap focuses the link before its click arrives, so the click alone cannot tell whether the node was already selected.
let pressedUnselected: SVGAElement | null = null;

document.addEventListener('pointerover', event => {
  const node = event.pointerType === 'touch' ? null : graphNode(event.target);
  if (node && !isSelected(node)) select(node);
});
document.addEventListener('focusin', event => {
  const node = graphNode(event.target);
  if (node && !isSelected(node)) select(node);
});
document.addEventListener('pointerdown', event => {
  const node = graphNode(event.target);
  pressedUnselected = node && !isSelected(node) ? node : null;
});
document.addEventListener('pointercancel', () => { pressedUnselected = null; });
document.addEventListener('click', event => {
  const node = graphNode(event.target), pressed = pressedUnselected;
  pressedUnselected = null;
  if (!node || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  // event.detail is 0 for a keyboard activation, which always follows focus and therefore a selection.
  if (isSelected(node) && !(event.detail && pressed === node)) return;
  event.preventDefault();
  select(node);
});
