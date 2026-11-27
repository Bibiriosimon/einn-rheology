import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STOPWORDS = {
    'the', 'and', 'or', 'for', 'with', 'under', 'into', 'from', 'that', 'this', 'are',
    'was', 'were', 'has', 'have', 'had', 'but', 'not', 'than', 'then', 'their', 'there',
    'about', 'through', 'between', 'where', 'which', 'while', 'when', 'into', 'onto',
    'very', 'more', 'most', 'less', 'much', 'many', 'does', 'doesn', 'than', 'such',
    'also', 'using', 'used', 'use', 'can', 'may', 'might', 'would', 'should', 'could',
    'model', 'models', 'equation', 'equations', 'polymer', 'rheology', 'viscosity'
}

TAG_HINTS = {
    'very_low_shear': {'shear_thinning', 'cross_model', 'carreau_yasuda', 'power_law', 'newtonian_plateau'},
    'low_shear': {'shear_thinning', 'cross_model', 'carreau_yasuda', 'power_law'},
    'high_phi': {'phi_effects', 'nanoparticle_loading', 'filler_loading', 'concentration_effects'},
    'mid_phi': {'phi_effects', 'concentration_effects'},
    'high_T': {'temperature_dependence', 'ttsp', 'wlf'},
    'mid_T': {'temperature_dependence', 'ttsp'},
}

NOISE_LAW_PAT = re.compile(r'no rheological content detected|introductory material|no substantive rheological content', re.I)
LAW_BONUS_PAT = re.compile(r'cross|carreau|yasuda|wlf|ttsp|maxwell|power law|shear thinning', re.I)
TOKEN_PAT = re.compile(r'[a-z0-9_\-\.]+')


def _tokenize(text: str) -> List[str]:
    toks = [t for t in TOKEN_PAT.findall((text or '').lower()) if len(t) > 1]
    return [t for t in toks if t not in STOPWORDS]


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def infer_regime(phi: float, logg: float, T: float) -> Dict[str, str]:
    if logg <= -3.0:
        shear = 'very_low_shear'
    elif logg <= -1.5:
        shear = 'low_shear'
    elif logg <= 0.5:
        shear = 'mid_shear'
    else:
        shear = 'high_shear'

    if phi >= 0.30:
        phi_level = 'high_phi'
    elif phi >= 0.12:
        phi_level = 'mid_phi'
    else:
        phi_level = 'low_phi'

    if T >= 2.0:
        temp = 'high_T'
    elif T >= 1.0:
        temp = 'mid_T'
    else:
        temp = 'low_T'

    return {'shear': shear, 'phi': phi_level, 'temp': temp}


def build_query_text(phi: float, logg: float, T: float, base_pred: float, prior_hint: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
    regime = infer_regime(phi, logg, T)
    parts = [
        f'phi {phi:.4f} {regime["phi"]}',
        f'log shear rate {logg:.4f} {regime["shear"]}',
        f'temperature {T:.4f} {regime["temp"]}',
        f'base prediction log viscosity {base_pred:.6f}',
        'focus on shear thinning temperature dependence and concentration effects',
    ]
    if prior_hint:
        direction = prior_hint.get('direction', 'uncertain')
        risk = prior_hint.get('risk', 'unknown')
        parts.append(f'correction direction hint {direction} with {risk} risk')
    return ' ; '.join(parts), regime


class ChunkKnowledgeBase:
    def __init__(self, knowledge_dir: Path, max_chunks: int = 0):
        self.knowledge_dir = Path(knowledge_dir)
        self.chunk_path = self.knowledge_dir / 'chunks_upgraded.jsonl'
        if not self.chunk_path.exists():
            raise FileNotFoundError(f'Knowledge chunks not found: {self.chunk_path}')
        self.records = []
        with self.chunk_path.open('r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if not self._keep_record(rec):
                    continue
                meta_text = ' '.join([
                    rec.get('law', ''),
                    rec.get('equation', ''),
                    rec.get('assumptions', ''),
                    rec.get('valid_range', ''),
                    ' '.join(rec.get('topic_tags', [])),
                    rec.get('text', ''),
                    rec.get('source_hint', ''),
                ])
                rec['_tokens'] = Counter(_tokenize(meta_text))
                rec['_token_count'] = sum(rec['_tokens'].values())
                rec['_has_law_bonus'] = bool(LAW_BONUS_PAT.search(rec.get('law', '') + ' ' + rec.get('equation', '')))
                self.records.append(rec)
                if max_chunks and len(self.records) >= max_chunks:
                    break
        if not self.records:
            raise RuntimeError('No usable rheology knowledge chunks were loaded.')

    @staticmethod
    def _keep_record(rec: Dict) -> bool:
        law = rec.get('law', '') or ''
        if NOISE_LAW_PAT.search(law):
            return False
        tags = rec.get('topic_tags', []) or []
        text = rec.get('text', '') or ''
        if not tags and len(text) < 80:
            return False
        return True

    def _score_record(self, rec: Dict, query_tokens: Counter, regime: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        overlap = 0.0
        if query_tokens:
            for tok, cnt in query_tokens.items():
                overlap += min(cnt, rec['_tokens'].get(tok, 0))
            overlap /= max(1.0, math.sqrt(sum(query_tokens.values()) * max(1, rec['_token_count'])))

        tags = set(rec.get('topic_tags', []) or [])
        tag_bonus = 0.0
        for regime_key in (regime['shear'], regime['phi'], regime['temp']):
            tag_bonus += 0.6 * len(tags & TAG_HINTS.get(regime_key, set()))

        law_bonus = 1.2 if rec['_has_law_bonus'] else 0.0
        structure_bonus = 0.2 * bool(rec.get('equation')) + 0.2 * bool(rec.get('assumptions')) + 0.2 * bool(rec.get('valid_range'))
        total = overlap + tag_bonus + law_bonus + structure_bonus
        return total, {
            'lexical': round(float(overlap), 6),
            'tag_bonus': round(float(tag_bonus), 6),
            'law_bonus': round(float(law_bonus), 6),
            'structure_bonus': round(float(structure_bonus), 6),
        }

    def retrieve(self, phi: float, logg: float, T: float, base_pred: float, prior_hint: Optional[Dict[str, str]] = None,
                 topk: int = 5, per_source_cap: int = 2) -> Tuple[str, Dict[str, str], List[Dict]]:
        query_text, regime = build_query_text(phi, logg, T, base_pred, prior_hint)
        query_tokens = Counter(_tokenize(query_text))
        candidates = []
        for rec in self.records:
            score, breakdown = self._score_record(rec, query_tokens, regime)
            if score <= 0.0:
                continue
            candidates.append((score, rec, breakdown))
        candidates.sort(key=lambda x: x[0], reverse=True)

        picked = []
        per_source = {}
        seen_ids = set()
        for score, rec, breakdown in candidates:
            src = rec.get('source_id', 'unknown')
            if rec.get('id') in seen_ids:
                continue
            if per_source.get(src, 0) >= per_source_cap:
                continue
            seen_ids.add(rec.get('id'))
            per_source[src] = per_source.get(src, 0) + 1
            picked.append({
                'id': rec.get('id'),
                'source_id': src,
                'book_title': rec.get('book_title', ''),
                'page_start': rec.get('page_start'),
                'page_end': rec.get('page_end'),
                'law': rec.get('law', ''),
                'equation': rec.get('equation', ''),
                'assumptions': rec.get('assumptions', ''),
                'valid_range': rec.get('valid_range', ''),
                'topic_tags': rec.get('topic_tags', []),
                'text': rec.get('text', ''),
                'source_hint': rec.get('source_hint', ''),
                'score': round(float(score), 6),
                'score_breakdown': breakdown,
            })
            if len(picked) >= topk:
                break
        return query_text, regime, picked


def format_evidence_text(evidence: List[Dict], max_text_chars: int = 600) -> str:
    blocks = []
    for rank, ev in enumerate(evidence, start=1):
        text = (ev.get('text') or '').replace('\n', ' ').strip()
        if len(text) > max_text_chars:
            text = text[:max_text_chars].rstrip() + ' ...'
        block = [
            f'Evidence {rank} | id={ev.get("id", "")}',
            f'Source: {ev.get("source_hint", ev.get("source_id", ""))}',
            f'Score: {ev.get("score", 0.0):.4f}',
            f'Tags: {", ".join(ev.get("topic_tags", []))}',
            f'Law: {ev.get("law", "")}',
        ]
        if ev.get('equation'):
            block.append(f'Equation: {ev.get("equation", "")}')
        if ev.get('assumptions'):
            block.append(f'Assumptions: {ev.get("assumptions", "")}')
        if ev.get('valid_range'):
            block.append(f'Valid range: {ev.get("valid_range", "")}')
        block.append(f'Text: {text}')
        blocks.append(' | '.join(block))
    return '\n'.join(blocks)


def extract_mechanisms(evidence: List[Dict], topn: int = 4) -> List[str]:
    counter = Counter()
    for ev in evidence:
        for tag in ev.get('topic_tags', []):
            if tag:
                counter[tag] += 1
        law = (ev.get('law') or '').strip()
        if law:
            short_law = law.split('.')[0][:120].strip()
            counter[short_law] += 1
    items = [k for k, _ in counter.most_common(topn)]
    return items or ['general rheology evidence']


def build_summary_struct(phi: float, logg: float, T: float, base_pred: float, regime: Dict[str, str],
                         evidence: List[Dict], prior_hint: Optional[Dict[str, str]] = None) -> Dict:
    mechanisms = extract_mechanisms(evidence)
    evidence_ids = [ev.get('id', '') for ev in evidence]
    if prior_hint and prior_hint.get('direction'):
        direction = prior_hint.get('direction', 'uncertain')
        strength = prior_hint.get('risk', 'medium')
    else:
        direction = 'uncertain'
        strength = 'medium' if evidence else 'low'

    rationale = []
    if regime['shear'] in ('very_low_shear', 'low_shear'):
        rationale.append('low-shear regime raises sensitivity to zero-shear plateau and shear-thinning onset')
    if regime['temp'] == 'high_T':
        rationale.append('high-temperature regime raises sensitivity to temperature dependence and shift-factor behavior')
    if regime['phi'] == 'high_phi':
        rationale.append('high-phi regime raises sensitivity to concentration-driven viscosity amplification')
    if mechanisms:
        rationale.append('retrieved evidence emphasizes ' + '; '.join(mechanisms[:3]))

    return {
        'regime_assessment': f"{regime['phi']}, {regime['shear']}, {regime['temp']}",
        'relevant_mechanism': mechanisms,
        'likely_bias_direction': direction,
        'correction_strength': strength,
        'evidence_backed_rationale': rationale,
        'evidence_ids': evidence_ids,
        'base_prediction': round(_safe_float(base_pred), 6),
        'state': {
            'phi': round(_safe_float(phi), 6),
            'logg': round(_safe_float(logg), 6),
            'T': round(_safe_float(T), 6),
        },
    }


def summary_struct_to_text(summary: Dict) -> str:
    return ' '.join([
        f"Regime assessment: {summary.get('regime_assessment', '')}.",
        'Relevant mechanisms: ' + '; '.join(summary.get('relevant_mechanism', [])) + '.',
        f"Likely bias direction: {summary.get('likely_bias_direction', 'uncertain')}.",
        f"Correction strength: {summary.get('correction_strength', 'medium')}.",
        'Rationale: ' + '; '.join(summary.get('evidence_backed_rationale', [])) + '.',
        'Evidence ids: ' + ', '.join(summary.get('evidence_ids', [])) + '.',
    ])
