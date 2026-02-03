# Sample Texts for Testing

This directory contains small sample texts for testing the polysemy analysis pipeline without requiring a large corpus.

## Contents

- `edo/sample_edo_text.txt` - Sample text in Edo period style (classical Japanese)
- `meiji/sample_meiji_text.txt` - Sample text in Meiji period style (modern Japanese)

## Purpose

These sample texts are provided for:
1. Quick validation of the pipeline functionality
2. Testing polysemy detection on known polysemous words
3. Ensuring the pipeline runs end-to-end without errors
4. Development and debugging

## Characteristics

### Edo Sample
- Written in classical Japanese style (文語 bungo)
- Uses classical grammar patterns (けり, なり, たり endings)
- Features target words: 世, 花, 心, 時, 人, 物, 所, 事, 手, 目, 日, 色
- Themes: impermanence (無常 mujō), nature, philosophical reflection
- Representative of Edo literature style

### Meiji Sample
- Written in modern Japanese style influenced by genbun itchi (言文一致)
- Uses modern grammar and sentence structures
- Features target words: 世, 花, 心, 時, 人, 物, 日, 色
- Themes: modernization, Western influence, individual consciousness
- Representative of Meiji literature style (similar to Natsume Sōseki's style)

## Usage

### Quick Test
```bash
# Run pipeline on sample data
python src/data_preprocess.py \
    --edo-dir data/samples/edo \
    --meiji-dir data/samples/meiji \
    --output data/processed_samples \
    --top-n 10 \
    --min-freq 2
```

### Expected Results

With these small samples, you should observe:
- Words like 心 (kokoro "heart/mind"), 世 (yo "world"), 花 (hana "flower") appearing in both eras
- Different contextual uses reflecting era-specific themes
- The Meiji sample showing more abstract/psychological uses of 心
- The Edo sample showing more traditional aesthetic uses of 花

## Limitations

These are synthetic samples created for testing purposes:
- **Not authentic historical texts** - created to demonstrate the pipeline
- **Limited vocabulary** - designed to include target words from target_words.csv
- **Small size** - not suitable for robust statistical analysis
- **Representative style only** - simplified versions of period-appropriate Japanese

## For Real Research

For actual research, use authentic texts from:
- Aozora Bunko (https://www.aozora.gr.jp/)
- National Institute for Japanese Language and Linguistics corpora
- University library digital collections

Place real corpus texts in:
- `data/edo/` for Edo period texts
- `data/meiji/` for Meiji period texts

## Notes on Word Usage in Samples

### 花 (hana)
- Edo sample: Traditional poetic usage ("花の盛り" - peak of blossoms)
- Meiji sample: Both literal ("花が咲き誇り") and metaphorical references

### 世 (yo)
- Edo sample: Buddhist/philosophical sense ("世の無常" - impermanence of the world)
- Meiji sample: Social/modern sense ("世の中" - society, "世間" - public sphere)

### 心 (kokoro)
- Edo sample: Traditional aesthetic sense ("心豊か" - rich in spirit)
- Meiji sample: Psychological sense ("心の奥底" - depths of the mind, "心の平安" - peace of mind)

### 時 (toki)
- Edo sample: Natural flow of time ("時の流れ" - flow of time)
- Meiji sample: Clock time, modern scheduling ("時計の音" - sound of clock)

These differences, while subtle in small samples, demonstrate the semantic shifts the pipeline aims to detect at scale.
