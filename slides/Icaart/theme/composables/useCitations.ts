import { ref, computed } from 'vue';

export interface BibEntry {
  key: string;
  author: string;
  title: string;
  booktitle?: string;
  journal?: string;
  year: string;
  url?: string;
  doi?: string;
}

const entries = ref(new Map<string, BibEntry>());
const slideCitations = new Map<number, string[]>();
/** Bump to trigger reactivity when citations are registered (Cite mounts after CiteFootnotes). */
const citationVersion = ref(0);
const loaded = ref(false);

function stripBraces(s: string): string {
  return s.replace(/\{\{|\}\}/g, '').replace(/\{[^}]*\}/g, (m) => m.slice(1, -1));
}

function stripLatex(s: string): string {
  return s
    .replace(/\{\\'e\}/g, 'e')
    .replace(/\{\\'a\}/g, 'a')
    .replace(/\{\\'i\}/g, 'i')
    .replace(/\{\\'o\}/g, 'o')
    .replace(/\{\\'u\}/g, 'u')
    .replace(/\{\\c{c}\}/gi, 'c')
    .replace(/\{\\u g\}/gi, 'g')
    .replace(/\{\\copyright\}/g, '©')
    .replace(/\{\\&}/g, '&')
    .replace(/\\&/g, '&')
    .replace(/\{\{[^}]*\}\}/g, stripBraces)
    .replace(/\{[^}]*\}/g, (m) => stripLatex(m.slice(1, -1)))
    .replace(/\s+/g, ' ')
    .trim();
}

function extractFieldValue(content: string, fieldName: string): string | undefined {
  const quotedRe =
    fieldName === 'title'
      ? /(?:^|[\s,])title\s*=\s*"([^"]*)"/i
      : new RegExp(`(?:^|[,\\s])${fieldName}\\s*=\\s*"([^"]*)"`, 'i');
  const quoted = content.match(quotedRe);
  if (quoted) return stripLatex(quoted[1]);

  const braceRe =
    fieldName === 'title'
      ? /(?:^|[\s,])title\s*=\s*\{/i
      : new RegExp(`(?:^|[,\\s])${fieldName}\\s*=\\s*\\{`, 'i');
  const braceMatch = content.match(braceRe);
  if (!braceMatch) return undefined;
  const braceStart = content.indexOf(braceMatch[0]);
  const startIdx = braceStart + braceMatch[0].length - 1;
  let depth = 0;
  let endIdx = -1;
  for (let i = startIdx; i < content.length; i++) {
    if (content[i] === '{') depth++;
    else if (content[i] === '}') {
      depth--;
      if (depth === 0) endIdx = i;
    }
  }
  const nextFieldRe = /\}\s*,\s*(?:\n\s*)?\w+\s*=/;
  const afterStart = content.slice(startIdx);
  const nextFieldMatch = afterStart.match(nextFieldRe);
  if (nextFieldMatch) {
    const boundaryEnd = startIdx + afterStart.indexOf(nextFieldMatch[0]);
    if (endIdx < 0 || boundaryEnd < endIdx) endIdx = boundaryEnd;
  }
  const raw = endIdx >= 0 ? content.slice(startIdx + 1, endIdx) : '';
  let cleaned = stripLatex(raw)
    .replace(/\s*\)\s*$/, '')
    .replace(/,?\s*url\s*=\s*.*$/i, '')
    .replace(/,?\s*year\s*=\s*\{?\s*.*$/i, '')
    .replace(/,?\s*author\s*=\s*\{?\s*.*$/i, '')
    .replace(/,?\s*booktitle\s*=\s*\{?\s*.*$/i, '')
    .trim();
  if (fieldName === 'title' && /^title\s*=\s*/i.test(cleaned)) {
    cleaned = cleaned.replace(/^title\s*=\s*/i, '').trim();
  }
  return cleaned;
}

function parseBibtex(raw: string): Map<string, BibEntry> {
  const result = new Map<string, BibEntry>();
  const entryRe = /@\s*\w+\s*\{\s*([^,\s]+)\s*,\s*([\s\S]*?)(?=@\s*\w+\s*\{|$)/g;
  let block: RegExpExecArray | null;
  while ((block = entryRe.exec(raw)) !== null) {
    const key = block[1].trim();
    const content = block[2];
    const author = extractFieldValue(content, 'author');
    const title = extractFieldValue(content, 'title');
    const booktitle = extractFieldValue(content, 'booktitle');
    const journal = extractFieldValue(content, 'journal');
    const year = extractFieldValue(content, 'year');
    const url = extractFieldValue(content, 'url');
    const doi = extractFieldValue(content, 'doi');
    if (author !== undefined && title !== undefined && year !== undefined) {
      result.set(key, {
        key,
        author: author || '',
        title: title || '',
        booktitle,
        journal,
        year: year || '',
        url,
        doi,
      });
    }
  }
  return result;
}

function getFirstAuthorSurname(author: string): string {
  const first = author.split(/\s+and\s+/i)[0]?.trim() || '';
  if (first.includes(',')) {
    return first.split(',')[0].trim() || first;
  }
  const parts = first.split(/\s+/).filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : first;
}

function abbreviateVenue(booktitle?: string, journal?: string): string {
  const v = (booktitle || journal || '').trim();
  if (!v) return '';
  const lower = v.toLowerCase();
  if (lower.includes('iclr')) return 'ICLR';
  if (lower.includes('neurips') || lower.includes('nips')) return 'NeurIPS';
  if (lower.includes('icml')) return 'ICML';
  if (lower.includes('cvpr')) return 'CVPR';
  if (lower.includes('eccv')) return 'ECCV';
  if (lower.includes('acl')) return 'ACL';
  if (lower.includes('emnlp')) return 'EMNLP';
  if (lower.includes('aaai')) return 'AAAI';
  if (lower.includes('ijcai')) return 'IJCAI';
  if (lower.includes('international conference')) {
    const m = v.match(/\{?([A-Z]{2,})\}?\s*\d{4}/) || v.match(/([A-Z]{2,})\s*\d{4}/);
    return m ? m[1].trim() : v;
  }
  return v.length > 40 ? v.slice(0, 37) + '...' : v;
}

export function formatAbbreviated(entry: BibEntry): string {
  const first = getFirstAuthorSurname(entry.author);
  const authorPart = first ? `${first} et al.` : '';
  const venue = abbreviateVenue(entry.booktitle, entry.journal);
  const venueAlreadyHasYear = venue && /\d{4}/.test(venue.trim());
  const venuePart = venue
    ? venueAlreadyHasYear ? venue.trim() : (entry.year ? `${venue} ${entry.year}` : venue)
    : entry.year;
  const parts = [authorPart, entry.title, venuePart].filter(Boolean);
  return parts.join(', ');
}

let loadResolver: (() => void) | null = null;
let globalLoadPromise: Promise<void> | null = null;

export function useCitations() {
  function loadBib(filename: string): Promise<void> {
    if (loaded.value && entries.value.size > 0) return Promise.resolve();
    if (globalLoadPromise) return globalLoadPromise;
    globalLoadPromise = new Promise((resolve) => {
      loadResolver = resolve;
    });
    fetch('/' + filename)
      .then((r) => r.text())
      .then((text) => {
        const parsed = parseBibtex(text);
        entries.value = new Map(parsed);
        loaded.value = true;
        loadResolver?.();
        loadResolver = null;
      })
      .catch(() => {
        loadResolver?.();
        loadResolver = null;
      });
    return globalLoadPromise;
  }

  function registerCitation(slideNo: number, refKey: string): number {
    let list = slideCitations.get(slideNo);
    if (!list) {
      list = [];
      slideCitations.set(slideNo, list);
    }
    const idx = list.indexOf(refKey);
    if (idx >= 0) return idx + 1;
    list.push(refKey);
    citationVersion.value += 1;
    return list.length;
  }

  function getSlideFootnotes(slideNo: number): Array<{ num: number; text: string; url?: string }> {
    citationVersion.value;
    const list = slideCitations.get(slideNo);
    if (!list || list.length === 0) return [];
    const map = entries.value;
    return list.map((key, i) => {
      const entry = map.get(key);
      const text = entry ? formatAbbreviated(entry) : `${key} (not found)`;
      const url = entry?.url ?? (entry?.doi ? `https://doi.org/${entry.doi}` : undefined);
      return { num: i + 1, text, url };
    });
  }

  return {
    entries: computed(() => entries.value),
    loaded,
    loadBib,
    registerCitation,
    getSlideFootnotes,
  };
}
