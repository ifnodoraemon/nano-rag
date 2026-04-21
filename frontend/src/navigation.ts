export type AppPage = 'validate' | 'investigate' | 'evaluate';
export type AudienceMode = 'simple' | 'expert';

const MODE_STORAGE_KEY = 'nano-rag-mode';

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

export function getStoredAudienceMode(): AudienceMode {
  if (typeof window === 'undefined') {
    return 'simple';
  }
  const raw = window.localStorage.getItem(MODE_STORAGE_KEY);
  return raw === 'expert' ? 'expert' : 'simple';
}

export function setAudienceMode(mode: AudienceMode): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(MODE_STORAGE_KEY, mode);
  window.dispatchEvent(new CustomEvent('nano-rag:set-mode', { detail: mode }));
}

export function navigateToPage(
  page: AppPage,
  anchorId?: string,
  audienceMode?: AudienceMode,
): void {
  if (audienceMode) {
    setAudienceMode(audienceMode);
  }
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
