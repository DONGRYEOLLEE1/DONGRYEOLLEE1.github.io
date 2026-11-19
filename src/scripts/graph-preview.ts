type Input = 'hovered' | 'focused';
type Selection = Record<Input, SVGAElement | null>;

const selections = new WeakMap<HTMLElement, Selection>();

function graphNode(target: EventTarget | null) {
  return target instanceof Element ? target.closest<SVGAElement>('.graph-node') : null;
}

function fillPreview(preview: HTMLElement, node: SVGAElement) {
  const title = preview.querySelector<HTMLElement>('.graph-preview-title')!;
  const date = preview.querySelector<HTMLTimeElement>('.graph-preview-date')!;
  title.textContent = node.getAttribute('aria-label');
  title.lang = node.getAttribute('lang') || '';
  date.dateTime = date.textContent = node.dataset.publishedAt || '';
}

export function measureRelatedDialogHeight(dialog: HTMLDialogElement) {
  const preview = dialog.querySelector<HTMLElement>('.graph-preview');
  let height = dialog.getBoundingClientRect().height;
  if (!preview) return height;

  // Reserve room when positioning the popup so a preview never moves the hovered node.
  preview.hidden = false;
  for (const node of dialog.querySelectorAll<SVGAElement>('.graph-node')) {
    fillPreview(preview, node);
    height = Math.max(height, dialog.getBoundingClientRect().height);
  }
  preview.hidden = true;
  return height;
}

function updatePreview(node: SVGAElement, input: Input, entering: boolean) {
  const graph = node.closest<HTMLElement>('.graph-wrap');
  const preview = graph?.querySelector<HTMLElement>('.graph-preview');
  if (!graph || !preview) return;

  const selection = selections.get(graph) ?? { hovered: null, focused: null };
  if (!entering && selection[input] !== node) return;
  selection[input] = entering ? node : null;
  selections.set(graph, selection);

  const active = entering ? node : selection[input === 'hovered' ? 'focused' : 'hovered'];
  if (active) fillPreview(preview, active);
  preview.hidden = !active;
}

function updatePointer(event: PointerEvent, entering: boolean) {
  if (event.pointerType === 'touch') return;
  const node = graphNode(event.target);
  if (node && node !== graphNode(event.relatedTarget)) updatePreview(node, 'hovered', entering);
}

document.addEventListener('pointerover', event => updatePointer(event, true));
document.addEventListener('pointerout', event => updatePointer(event, false));
document.addEventListener('focusin', event => {
  const node = graphNode(event.target);
  if (node?.matches(':focus-visible')) updatePreview(node, 'focused', true);
});
document.addEventListener('focusout', event => {
  const node = graphNode(event.target);
  if (node) updatePreview(node, 'focused', false);
});
