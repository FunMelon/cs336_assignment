use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

type TokenId = usize;
type SeqId = usize;
type PairKey = (TokenId, TokenId);

#[derive(Clone)]
struct Seq {
    tokens: Vec<TokenId>,
    freq: u32,
    active: bool,
}

#[derive(Clone)]
struct HeapItem {
    freq: u64,
    a: TokenId,
    b: TokenId,
    a_bytes: Arc<[u8]>,
    b_bytes: Arc<[u8]>,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.freq == other.freq && self.a == other.a && self.b == other.b
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.freq.cmp(&other.freq) {
            Ordering::Equal => match self.a_bytes.as_ref().cmp(other.a_bytes.as_ref()) {
                Ordering::Equal => self.b_bytes.as_ref().cmp(other.b_bytes.as_ref()),
                ord => ord,
            },
            ord => ord,
        }
    }
}

fn pair_counts(seq: &[TokenId]) -> HashMap<PairKey, u32> {
    let mut counts: HashMap<PairKey, u32> = HashMap::new();
    if seq.len() < 2 {
        return counts;
    }
    for w in seq.windows(2) {
        let pair = (w[0], w[1]);
        *counts.entry(pair).or_insert(0) += 1;
    }
    counts
}

fn merge_pair_in_seq(seq: &[TokenId], pair: PairKey, new_id: TokenId) -> Vec<TokenId> {
    if seq.len() < 2 {
        return seq.to_vec();
    }
    let (a, b) = pair;
    let mut merged: Vec<TokenId> = Vec::with_capacity(seq.len());
    let mut i = 0usize;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == a && seq[i + 1] == b {
            merged.push(new_id);
            i += 2;
        } else {
            merged.push(seq[i]);
            i += 1;
        }
    }
    merged
}

fn push_pair(
    heap: &mut BinaryHeap<HeapItem>,
    pair_freq: &HashMap<PairKey, u64>,
    tokens: &[Arc<[u8]>],
    pair: PairKey,
) {
    let freq = pair_freq.get(&pair).copied().unwrap_or(0);
    if freq == 0 {
        return;
    }
    let (a, b) = pair;
    heap.push(HeapItem {
        freq,
        a,
        b,
        a_bytes: tokens[a].clone(),
        b_bytes: tokens[b].clone(),
    });
}

#[pyfunction]
fn bpe_merge_cached(
    py: Python,
    seqs_and_freqs: Vec<(Vec<Vec<u8>>, u32)>,
    vocab: Vec<Vec<u8>>,
    vocab_size: usize,
) -> PyResult<(Vec<Py<PyBytes>>, Vec<(Py<PyBytes>, Py<PyBytes>)>)> {
    // Token table: token_id -> bytes
    let mut tokens: Vec<Arc<[u8]>> = Vec::with_capacity(vocab.len().max(vocab_size));
    let mut token2id: HashMap<Vec<u8>, TokenId> = HashMap::new();

    for (i, tok) in vocab.into_iter().enumerate() {
        token2id.insert(tok.clone(), i);
        tokens.push(Arc::<[u8]>::from(tok));
    }

    // Collapse identical sequences by summing frequencies.
    let mut seq_freq_map: HashMap<Vec<TokenId>, u32> = HashMap::new();
    for (seq_tokens, freq) in seqs_and_freqs.into_iter() {
        if freq == 0 {
            continue;
        }
        let mut seq_ids: Vec<TokenId> = Vec::with_capacity(seq_tokens.len());
        for t in seq_tokens.into_iter() {
            let tid = match token2id.get(&t) {
                Some(&id) => id,
                None => {
                    let id = tokens.len();
                    token2id.insert(t.clone(), id);
                    tokens.push(Arc::<[u8]>::from(t));
                    id
                }
            };
            seq_ids.push(tid);
        }
        *seq_freq_map.entry(seq_ids).or_insert(0) += freq;
    }

    let mut seq_map: HashMap<Vec<TokenId>, SeqId> = HashMap::new();
    let mut sequences: Vec<Seq> = Vec::with_capacity(seq_freq_map.len());
    for (seq_ids, freq) in seq_freq_map.into_iter() {
        let sid = sequences.len();
        seq_map.insert(seq_ids.clone(), sid);
        sequences.push(Seq {
            tokens: seq_ids,
            freq,
            active: true,
        });
    }

    // Global pair frequencies and inverted index.
    let mut pair_freq: HashMap<PairKey, u64> = HashMap::new();
    let mut pair2seqs: HashMap<PairKey, HashSet<SeqId>> = HashMap::new();
    for (sid, seq) in sequences.iter().enumerate() {
        if !seq.active || seq.tokens.len() < 2 {
            continue;
        }
        let f = seq.freq as u64;
        for w in seq.tokens.windows(2) {
            let pair = (w[0], w[1]);
            *pair_freq.entry(pair).or_insert(0) += f;
            pair2seqs.entry(pair).or_default().insert(sid);
        }
    }

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for (&pair, &freq) in pair_freq.iter() {
        if freq == 0 {
            continue;
        }
        let (a, b) = pair;
        heap.push(HeapItem {
            freq,
            a,
            b,
            a_bytes: tokens[a].clone(),
            b_bytes: tokens[b].clone(),
        });
    }

    let mut merges_ids: Vec<PairKey> = Vec::new();

    while tokens.len() < vocab_size {
        // Pop best current pair (lazy deletion).
        let mut best_pair: Option<PairKey> = None;
        while let Some(item) = heap.pop() {
            let pair = (item.a, item.b);
            let cur = pair_freq.get(&pair).copied().unwrap_or(0);
            if cur != item.freq {
                continue;
            }
            if cur == 0 {
                continue;
            }
            best_pair = Some(pair);
            break;
        }

        let Some(pair) = best_pair else {
            break;
        };

        let (a, b) = pair;
        let mut new_bytes: Vec<u8> = Vec::with_capacity(tokens[a].len() + tokens[b].len());
        new_bytes.extend_from_slice(tokens[a].as_ref());
        new_bytes.extend_from_slice(tokens[b].as_ref());

        let new_id = tokens.len();
        tokens.push(Arc::<[u8]>::from(new_bytes));
        merges_ids.push(pair);

        let affected: Vec<SeqId> = match pair2seqs.remove(&pair) {
            Some(set) => set.into_iter().collect(),
            None => {
                // Should be rare; keep going.
                pair_freq.remove(&pair);
                continue;
            }
        };

        for sid in affected.into_iter() {
            if sid >= sequences.len() {
                continue;
            }
            if !sequences[sid].active {
                continue;
            }

            let old_seq = sequences[sid].tokens.clone();
            let old_freq = sequences[sid].freq;
            if old_seq.len() < 2 {
                continue;
            }

            let old_counts = pair_counts(&old_seq);
            let new_seq = merge_pair_in_seq(&old_seq, pair, new_id);
            if new_seq == old_seq {
                continue;
            }
            let new_counts = pair_counts(&new_seq);

            // Remove old seq key from map.
            seq_map.remove(&old_seq);

            // Decrement global pair frequencies and remove inverted index refs.
            for (&p, &cnt) in old_counts.iter() {
                let dec = (old_freq as u64) * (cnt as u64);
                if let Some(v) = pair_freq.get_mut(&p) {
                    *v = v.saturating_sub(dec);
                    if *v == 0 {
                        pair_freq.remove(&p);
                    }
                }

                if let Some(set) = pair2seqs.get_mut(&p) {
                    set.remove(&sid);
                    if set.is_empty() {
                        pair2seqs.remove(&p);
                    }
                }
            }

            // If a sequence with identical new tokens exists, merge frequencies into it.
            if let Some(&existing_id) = seq_map.get(&new_seq) {
                sequences[existing_id].freq = sequences[existing_id].freq.saturating_add(old_freq);

                for (&p, &cnt) in new_counts.iter() {
                    let inc = (old_freq as u64) * (cnt as u64);
                    *pair_freq.entry(p).or_insert(0) += inc;
                    pair2seqs.entry(p).or_default().insert(existing_id);
                }

                sequences[sid].active = false;
                sequences[sid].freq = 0;
                sequences[sid].tokens.clear();
            } else {
                sequences[sid].tokens = new_seq.clone();
                seq_map.insert(new_seq, sid);

                for (&p, &cnt) in new_counts.iter() {
                    let inc = (old_freq as u64) * (cnt as u64);
                    *pair_freq.entry(p).or_insert(0) += inc;
                    pair2seqs.entry(p).or_default().insert(sid);
                }
            }

            // Re-push affected pairs into heap (lazy, may include stale items).
            let mut affected_pairs: HashSet<PairKey> = HashSet::new();
            for (&p, _) in old_counts.iter() {
                affected_pairs.insert(p);
            }
            for (&p, _) in new_counts.iter() {
                affected_pairs.insert(p);
            }
            for p in affected_pairs.into_iter() {
                push_pair(&mut heap, &pair_freq, &tokens, p);
            }
        }
    }

    // Build Python outputs.
    let vocab_out: Vec<Py<PyBytes>> = tokens
        .iter()
        .map(|t| PyBytes::new_bound(py, t.as_ref()).unbind())
        .collect();

    let merges_out: Vec<(Py<PyBytes>, Py<PyBytes>)> = merges_ids
        .into_iter()
        .map(|(a, b)| {
            (
                PyBytes::new_bound(py, tokens[a].as_ref()).unbind(),
                PyBytes::new_bound(py, tokens[b].as_ref()).unbind(),
            )
        })
        .collect();

    Ok((vocab_out, merges_out))
}

#[pymodule]
fn _bpe_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bpe_merge_cached, m)?)?;
    Ok(())
}
