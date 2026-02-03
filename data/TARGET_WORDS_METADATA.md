# Target Words Metadata

This file documents the target words selected for polysemy analysis and provides linguistic context for interpreting results.

## Word Selection Criteria

The words in `target_words.csv` were selected based on:

1. **High Frequency**: Common in both Edo and Meiji literature
2. **Known Polysemy**: Multiple established senses in Japanese dictionaries
3. **Historical Relevance**: Likely to show semantic change during modernization
4. **Content Word Status**: Nouns with substantive meanings (not grammatical particles)

## Metadata Fields

- **word**: Japanese kanji form
- **reading**: Hiragana reading (primary pronunciation)
- **pos**: Part of speech (名詞 = noun, 動詞 = verb, etc.)
- **known_senses**: Semicolon-separated list of established meanings
- **edo_notes**: Context about usage patterns in Edo period (1603-1868)
- **meiji_notes**: Expected changes or new meanings in Meiji period (1868-1912)
- **polysemy_expectation**: Predicted direction of change
  - "Expected increase": Anticipate more senses in Meiji
  - "Moderate change": Some expansion or specialization likely
  - "Minimal change": Core meanings relatively stable

## Example Words of Interest

### 花 (はな, hana)
**Classical meanings**: flower, blossom
**Extended meanings**: nose (from shape), peak/prime (metaphorical)
**Meiji context**: Western botanical terminology (species names, scientific classification) may have influenced usage patterns

### 世 (よ, yo)
**Classical meanings**: generation, age, era, world
**Edo usage**: Often in literary/philosophical contexts (浮世 ukiyo "floating world")
**Meiji context**: Modern concepts of "society" (社会 shakai), "public sphere," Western social science terminology

### 心 (こころ, kokoro)
**Classical meanings**: heart, mind, spirit, essence, feeling
**Edo usage**: Deeply embedded in aesthetic discourse (mono no aware, aware)
**Meiji context**: Introduction of Western psychology (心理学 shinrigaku), concepts like "consciousness" (意識 ishiki) may have split semantic space

## Data Normalization Notes

**Important considerations for accurate analysis:**

1. **Orthographic Variation**:
   - Edo texts often use historical kana (旧仮名遣い kyū kanazukai)
   - Variant kanji forms (異体字 itaiji) were more common in Edo period
   - **Recommendation**: Normalize historical orthography to modern equivalents before tokenization

2. **Loanword Status**:
   - Mark gairaigo (外来語 loanwords) separately
   - Many Meiji semantic expansions come from new compounds (e.g., 電話 denwa "telephone") rather than polysemy of native words
   - **Filter**: Focus analysis on yamato kotoba (native Japanese) and kango (Sino-Japanese) established before 1600

3. **Genre Effects**:
   - Edo corpus: gesaku (戯作 playful fiction), kabuki (歌舞伎), haikai (俳諧 poetry)
   - Meiji corpus: shōsetsu (小説 modern novel), journalism, translated works
   - **Control variable**: Track genre distribution and consider stratified analysis

4. **Register/Style**:
   - Edo: Classical literary Japanese with heavy Chinese influence in some genres
   - Meiji: Genbun itchi (言文一致 unification of written and spoken language) movement created new written styles
   - **Impact**: Apparent polysemy changes may reflect stylistic shifts rather than semantic expansion

## Manual Sense Annotations (Planned)

For validation, we recommend manually annotating a small subset:
- Select 3-5 high-priority words (e.g., 花, 世, 心)
- Extract 20-50 random contexts per era
- Label with sense tags based on dictionary
- Compare manual annotations with clustering results
- Calculate adjusted Rand index or similar metric to validate automated sense detection

## Baseline Polysemy Sources (Future Work)

Potential dictionary sources for baseline comparison:
1. **Kōjien** (広辞苑): Authoritative modern Japanese dictionary
2. **Nihon Kokugo Daijiten** (日本国語大辞典): Historical dictionary with dated attestations
3. **Iwanami Kogo Jiten** (岩波古語辞典): Classical Japanese dictionary for Edo baseline
4. **Digital Daijisen API**: Programmatic access to dictionary data

**Approach**: 
- Count dictionary senses per word as ground truth
- Compare Edo vs. Meiji polysemy_index to dictionary sense counts
- Identify words where computational clusters align with/diverge from lexicographic distinctions

## Sensitivity Analysis Recommendations

1. **Era Boundaries**:
   - Test with sliding windows: 1800-1850 vs 1870-1920
   - Narrow period: 1860-1875 (immediate pre/post Restoration)
   - Control for author birth cohort

2. **Cluster Parameters**:
   - Vary k-means k range (currently 2-10, try 2-15)
   - Test DBSCAN with different epsilon values
   - Compare hierarchical clustering with different linkage methods

3. **Embedding Aggregation**:
   - Current: average subword embeddings
   - Alternative: CLS token only, weighted average, attention-based pooling
   - Effect: May change cluster structure

## References

- Tamaoka, K., & Altmann, G. (2018). "Polysemy in Japanese: Effects of semantic similarity on acceptability judgments." _Journal of Psycholinguistic Research_, 47(2), 425-441.
- Traugott, E. C., & Dasher, R. B. (2001). _Regularity in Semantic Change_. Cambridge University Press.
- Historical Corpus of Japanese: https://clrd.ninjal.ac.jp/chj/

## Contact

For questions about word selection or metadata, open an issue on GitHub.
