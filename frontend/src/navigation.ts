export type AppPage = 'validate' | 'investigate' | 'evaluate';

const PAGE_HASHES: Record<AppPage, string> = {
  validate: '#/validate',
  investigate: '#/investigate',
  evaluate: '#/evaluate',
};

export function getPageFromHash(hash: string): AppPage {
  if (hash === PAGE_HASHES.investigate) {
    return 'investigate';
  }
  if (hash === PAGE_HASHES.evaluate) {
    return 'evaluate';
  }
  return 'validate';
}

export function navigateToPage(page: AppPage, anchorId?: string): void {
  window.location.hash = PAGE_HASHES[page];
  if (!anchorId) {
    return;
  }
  window.setTimeout(() => {
    document.getElementById(anchorId)?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  }, 80);
}
